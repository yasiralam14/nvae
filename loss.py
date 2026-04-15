import torch
import torch.nn as nn
import torch.nn.functional as F


class Nvae_Loss(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.pad_idx = hyperparameters["bart_pad_id"]
        self.local_latent_dim = hyperparameters["local_latent_dim"]

        self.cross_entropy_loss = nn.CrossEntropyLoss(
            ignore_index=self.pad_idx, reduction="none"
        )

        self.free_bits_global = 0.5
        self.free_bits_local = 0.5

    def forward(
    self,
    mu_t_q, sigma2_t_q,
    mu_i_q, sigma2_i_q,
    mu_i_p, sigma2_i_p,
    reconstruction_logits,
    target_ids, word_mask,
    local_kl_beta=0.5, global_kl_beta=0.1,
    sr_loss=0.0, sr_lambda=0.01
    ):
        eps = 1e-9
        B, W = target_ids.shape
        
        
        shift_logits = reconstruction_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, reconstruction_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=self.pad_idx,
            reduction='none'
        )
        
        per_token_loss = per_token_loss.view(B, -1)
        
        reconstruction_loss = per_token_loss.sum(dim=1).mean()

       
        shift_mask = (shift_labels != self.pad_idx).float()
        num_active_tokens = shift_mask.sum()
        mean_token_loss = per_token_loss.sum() / (num_active_tokens + eps)


        
        global_kl_raw = 0.5 * (
            sigma2_t_q + mu_t_q.pow(2) - 1.0 - torch.log(sigma2_t_q + eps)
        )

        
        global_frac_dims_clamped = (global_kl_raw < self.free_bits_global).float().mean()
        
        global_kl_charged = torch.clamp(global_kl_raw, min=self.free_bits_global)

        global_kl_charged_per_ex = global_kl_charged.sum(dim=-1)
        global_kl_raw_mean = global_kl_raw.sum(dim=-1).mean()
        
        global_kl_loss = global_kl_charged_per_ex.mean()


        mu_p_flat = mu_i_p.reshape(-1, self.local_latent_dim)
        sigma2_p_flat = sigma2_i_p.reshape(-1, self.local_latent_dim)
        mu_q_flat = mu_i_q.reshape(-1, self.local_latent_dim)
        sigma2_q_flat = sigma2_i_q.reshape(-1, self.local_latent_dim)

        term1 = torch.log(sigma2_p_flat + eps) - torch.log(sigma2_q_flat + eps)
        term2 = (sigma2_q_flat + (mu_q_flat - mu_p_flat).pow(2)) / (sigma2_p_flat + eps)
        local_kl_raw_flat = 0.5 * (term1 + term2 - 1.0)

        local_kl_raw = local_kl_raw_flat.view(B, W, self.local_latent_dim)

        mask_float = word_mask.float()
        
        local_under = (local_kl_raw < self.free_bits_local).float()
        valid_elements = mask_float.sum() * self.local_latent_dim
        local_frac_dims_clamped = (local_under * mask_float.unsqueeze(-1)).sum() / (valid_elements + eps)

        local_kl_charged = torch.clamp(local_kl_raw, min=self.free_bits_local)

        local_kl_raw_per_token = local_kl_raw.sum(dim=-1)
        local_kl_charged_per_token = local_kl_charged.sum(dim=-1)

        local_kl_raw_per_ex = (local_kl_raw_per_token * mask_float).sum(dim=1)
        local_kl_charged_per_ex = (local_kl_charged_per_token * mask_float).sum(dim=1)

        local_kl_raw_mean = local_kl_raw_per_ex.mean()
        local_kl_loss = local_kl_charged_per_ex.mean()


        total_loss = (
            reconstruction_loss
            + global_kl_beta * global_kl_loss
            + local_kl_beta * local_kl_loss
        )

        total_kl = global_kl_beta * global_kl_loss + local_kl_beta * local_kl_loss
        kl_ratio = (reconstruction_loss / (total_kl + 1e-8)).detach()
        
        loss_dict = {
            'total_loss': total_loss,
            'reconstruction_loss' : reconstruction_loss,
            'global_kl_loss' : global_kl_loss,
            'local_kl_loss' : local_kl_loss,
            'kl_ratio' : kl_ratio,
            'mean_token_loss' : mean_token_loss,
            'global_kl_raw_mean' : global_kl_raw_mean,
            'local_kl_raw_mean' : local_kl_raw_mean,
            'global_frac_dims_clamped' : global_frac_dims_clamped,
            'local_frac_dims_clamped' : local_frac_dims_clamped
        }

        return loss_dict