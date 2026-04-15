# NVAE-BART — Hierarchical Text VAE with Residual Shifters

> A PyTorch implementation of a Hierarchical Variational Autoencoder (HVAE) for text generation, built on top of the **BART** (Bidirectional and Auto-Regressive Transformers) architecture. This model uses a deeply integrated hierarchical latent space with spectral normalisation, residual shifting, and dynamic loss balancing.

---

## Overview

Generating coherent and well-structured text requires models that can capture both high-level semantics (the overall theme of a sequence) and low-level syntax (token-by-token choices). **NVAE-BART** learns a two-level hierarchy of latent variables grafted into the BART encoder-decoder framework:

| Latent | Represents | Mechanism |
|---|---|---|
| Global (`g_mu`, `g_sigma2`) | Sentence/Sequence level meaning | Derived from the pooled representations of BART encoder outputs |
| Local (`l_post_mu`, `l_post_sigma2`) | Token-level nuances | Token-by-token autoregressive prior and posterior |

At decoding, the global latent is projected into "soft tokens" and fed into the BART decoder. The token-level local latents are inferred using **Residual Shifters** (inspired by NVAE: A Deep Hierarchical Variational Autoencoder), which predict offsets relative to an autoregressive prior.

---

## Architecture

```
Input Sequence
       │
       ▼
┌────────────────────────────────────────────────────────┐
│           Inference Network  q(z | x)                  │
│                                                        │
│  BART Encoder                                          │
│    └── local_features (seq_len, hidden)                │
│                                                        │
│  Global Shift:                                         │
│    local_features (pooled) → global_shifter → z_global │
│                                                        │
│  Local Shift:                                          │
│    BART Decoder (fed with z_global soft tokens)        │
│    └── top_down_features (seq_len, hidden)             │
│                                                        │
│    [local_features; top_down_features]                 │
│         ↓                                              │
│    local_shifter →  shift relative to prior → z_local  │
└────────────────────────────────────────────────────────┘
       │                │
       ▼                ▼
    z_global ~ q     z_local ~ q
       │                │
       └────────┬───────┘
                ▼
┌────────────────────────────────────────────────────────┐
│           Generative Network  p(x | z)                 │
│                                                        │
│  z_local → latent_to_bart_mlp → Projected Z            │
│         ↓                                              │
│  Projected Z → F.linear(bart.shared.weight) → Logits   │
└────────────────────────────────────────────────────────┘
```

### Key Design Choices

- **Residual Shifters** — Rather than predicting the absolute mean and variance of the local latents, the model predicts the *delta* (shift) relative to a dynamically computed prior.
- **Spectral Normalization** — To stabilize the highly complex residual paths in the VAE, residual convolutions/linear layers are bounded using Spectral Normalization (`spectral_norm`).
- **Soft Tokens Injection** — The global latent variable is projected via an MLP into `k=10` soft tokens before being passed as `encoder_hidden_states` to the BART decoder.
- **Word Dropout / Masking** — To prevent the decoder from ignoring the latent code, a configurable masking probability (`mask_prob`) dynamically masks local features during training.
- **Dynamic Spectral Regularization** — The training loop computes a fast spectral radius loss (`compute_sr_loss`) via power iteration on the weights of the shift networks to enforce Lipschitz constraints.

---

## Repository Structure

```
nvae/
├── model.py            # BartHVAE, ResidualShifter, GlobalToSoftTokens
├── loss.py             # Nvae_Loss (Standard free-bits KL + Reconstruction)
├── dynamic_loss.py     # Nvae_Loss_Dynamic (Loss with Spectral Regularization and KL balancing)
├── train.py            # Lightning-fast PyTorch dataloaders and training loop
├── get_kl_beta.py      # KL-Annealing Schedule (Cyclical / Monotonic)
├── warmup_beta.py      # Additional Warmup logic
├── logs/               # Tensorboard / local logs
└── dynamic_hvae_checkpoints/ # Output directory for models (gitignored)
```

---

## Dependencies & Requirements

- `torch`, `torchvision`, `torchaudio`
- `transformers` (HuggingFace, for `BART-base`)
- `sklearn`, `pandas`
- `wandb` (for experiment tracking)
- `tqdm`

Install all dependencies via pip:
```bash
pip install torch transformers scikit-learn pandas wandb tqdm
```

---

## Usage

### Training

To begin training the model from scratch on the target dataset:

```bash
cd nvae
python train.py
```

`train.py` contains settings for the `WandB` project (`nvae_text_0_threshold`). Ensure you update the API key or authenticate locally using `wandb login`.

**Training Features:**
- Uses lazy tokenization and cleaning via `LazyTextDataset` to minimize RAM overhead.
- Implements KL beta scheduled annealing (`get_kl_beta()`) for smooth posterior alignment.
- Computes spectral radius loss dynamically (`compute_sr_loss()`) using power iterations.

### Inference & Generation

The `BartHVAE` module exposes two methods for interaction at test time:
- `reconstruct(input_ids, attention_mask)`: Embeds a sequence into the hierarchy and autoregressively reconstructs it.
- `generate(max_length=50, device='cpu')`: Samples random noise for `z_global` and autoregressively generates text unconditionally by sampling local priors.

---

## Loss Components

1. **Reconstruction Loss**: Standard cross-entropy evaluated over the non-pad tokens.
2. **Global KL Divergence**: Enforces that $q(z_{\text{global}} | x)$ approximates $\mathcal{N}(0, I)$.
3. **Local KL Divergence**: Enforces that $q(z_{\text{local}} | x, z_{\text{global}})$ approximates the predicted prior $p(z_{\text{local}} | z_{\text{global}}, z_{<t})$.
4. **Spectral Radius Loss (SR Reg)**: Enforces spectral bounds on the weight matrices of the local/global shifters to guarantee contractive mappings, thereby maintaining stability during MCMC or generation.

---

## References

- Vahdat & Kautz. *NVAE: A Deep Hierarchical Variational Autoencoder.* NeurIPS 2020.  
- Lewis et al. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation.* ACL 2020.  
- Bowman et al. *Generating Sentences from a Continuous Space.* CoNLL 2016.

---

## License

MIT License. See `LICENSE` for details.
