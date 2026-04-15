import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from transformers import BartTokenizer
from .model import BartHVAE
from .loss import Nvae_Loss
from .dynamic_loss import Nvae_Loss_Dynamic
from .get_kl_beta import get_kl_beta
import wandb
import pandas as pd
import os
import glob
from .warmup_beta import get_kl_beta_and_warmup_flag
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F




import gc
from torch.utils.data import Dataset, DataLoader

class LazyTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=50):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def strip_bos_eos(self, text: str) -> str:
        text = text.replace("<BOS>", "").replace("<EOS>", "")
        text = text.replace("</s>", " [EOS] ").replace("<s>", " [BOS] ")
        return " ".join(text.split()).strip()

    def __getitem__(self, idx):
        # Clean the text lazily
        clean_text = self.strip_bos_eos(self.texts[idx])
        
        # Tokenize lazily
        encoding = self.tokenizer(
            clean_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)


def compute_sr_loss(model, n_power_iterations=1):
    sr_loss = 0.0
    
    # Strictly target the VAE residual and prior networks
    target_networks = [
        model.global_shifter,
        model.local_shifter,
        model.local_prior_net
    ]
    
    for net in target_networks:
        for module in net.modules():
            if isinstance(module, nn.Linear):
                w = module.weight
                w_mat = w.view(w.size(0), -1)

                if not hasattr(module, 'u_vector'):
                    u_init = F.normalize(torch.randn(w_mat.size(0), 1, device=w.device), dim=0)
                    v_init = F.normalize(torch.randn(w_mat.size(1), 1, device=w.device), dim=0)
                    module.register_buffer('u_vector', u_init)
                    module.register_buffer('v_vector', v_init)

                u = module.u_vector
                v = module.v_vector

                with torch.no_grad():
                    for _ in range(n_power_iterations):
                        v = F.normalize(torch.mm(w_mat.t(), u), dim=0)
                        u = F.normalize(torch.mm(w_mat, v), dim=0)
                    
                    module.u_vector.copy_(u)
                    module.v_vector.copy_(v)

                sigma = torch.mm(u.t(), torch.mm(w_mat, v)).squeeze()
                sr_loss += sigma
                
    return sr_loss
wandb.login(key="0ce56922c7ea30310a87d49246b15bc7d7ca9c89")

hyperParams = {}
wandb.init(
    project="nvae_text_0_threshold",
    name = 'with_sr_reg_equilibrium'
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
df = pd.read_parquet("/home/salam4/hvae_project/word_hvae/ht_hvae/HVAE/data/sentences_df.parquet")
def strip_bos_eos(text: str) -> str:
    text = text.replace("<BOS>", "").replace("<EOS>", "")
    text = text.replace("</s>", " [EOS] ").replace("<s>", " [BOS] ")
    return " ".join(text.split()).strip()

raw_texts = df['sentence'].tolist()
train_texts, test_texts = train_test_split(raw_texts, test_size=0.0001, random_state=42)    
train_texts, val_texts = train_test_split(train_texts, test_size=0.001, random_state=42)
train_dataset = LazyTextDataset(train_texts, tokenizer)
val_dataset = LazyTextDataset(val_texts, tokenizer)

train_loader = DataLoader(
    train_dataset, 
    batch_size=256, 
    shuffle=True,       
    num_workers=4,      
    pin_memory=False    
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=64, 
    shuffle=False      
)

del df 
gc.collect() 


print(f"BOS Token ID (<s>):    {tokenizer.bos_token_id}")
print(f"PAD Token ID (<pad>):  {tokenizer.pad_token_id}")
print(f"EOS Token ID (</s>):   {tokenizer.eos_token_id}")

hyperParams['bart_pad_id'] = tokenizer.pad_token_id
hyperParams['local_latent_dim'] = 32
hyperParams['global_latent_dim'] = 768
hyperParams['model_name'] = 'facebook/bart-base'
hyperParams['mask_prob'] = 0


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BartHVAE(hyperParams).to(device)
    loss_module = Nvae_Loss_Dynamic(hyperParams)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    steps_per_epoch = len(train_loader)
    save_dir = "./dynamic_hvae_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    keep_last_n = 3
        
    model.train()
    for epoch in range(5):
        is_warmup = epoch < 0
        print(f"Epoch {epoch}")
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        
            kl_beta, _ = get_kl_beta(
                epoch,
                batch_idx,
                steps_per_epoch,
                MIN_BETA=0.001,  
                MAX_BETA=1,    
                CYCLE_EPOCHS=1, 
                MIN_HOLD_FRAC=0.5, 
                RAMP_FRAC=0.25   
            )
        
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
        
            outputs = model(input_ids, attention_mask)
            
            mu_t_q = outputs['g_mu']
            sigma2_t_q = outputs['g_sigma2']
            mu_i_q = outputs['l_post_mu']
            sigma2_i_q = outputs['l_post_sigma2']
            mu_i_p = outputs['l_prior_mu']
            sigma2_i_p = outputs['l_prior_sigma2']
            reconstruction_logits = outputs['logits']
            prior_logits = outputs['logits_prior']
            
            sr_loss = compute_sr_loss(model)
            
            loss_dict = loss_module(
                mu_t_q, sigma2_t_q,
                mu_i_q, sigma2_i_q,
                mu_i_p, sigma2_i_p,
                reconstruction_logits,
                prior_logits,
                input_ids, attention_mask,
                local_kl_beta=kl_beta, global_kl_beta=kl_beta,
                is_warmup=is_warmup,
                sr_loss = sr_loss,
                sr_lambda = 10
            )
            loss = loss_dict['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if batch_idx % 100 == 0:
                recon = loss_dict['reconstruction_loss']
                kl_ratio = loss_dict['kl_ratio']
                global_kl_loss = loss_dict['global_kl_loss']
                local_kl_loss = loss_dict['local_kl_loss']
                mean_token_loss = loss_dict['mean_token_loss']
                global_kl_raw_mean = loss_dict['global_kl_raw_mean']
                local_kl_raw_mean = loss_dict['local_kl_raw_mean']
                global_frac_dims_clamped = loss_dict['global_frac_dims_clamped']
                local_frac_dims_clamped = loss_dict['local_frac_dims_clamped']
                gamma_global = loss_dict['gamma_global']
                gamma_local = loss_dict['gamma_local']
                average_tokens = loss_dict['average_tokens']
                prior_variance_mean = loss_dict['prior_variance_mean']
                mean_token_loss_prior = loss_dict['mean_token_loss_prior']
                log_dict = {
                    "KL/kl_global_raw": global_kl_raw_mean.item(),
                    "KL/kl_local_raw": local_kl_raw_mean.item(),
                    "KL/kl_global_clamp_frac": global_frac_dims_clamped.item(),
                    "KL/kl_local_clamp_frac": local_frac_dims_clamped.item(),
                    "Losses/per_token_loss": mean_token_loss.item(),
                    "Losses/mean_token_loss_prior":mean_token_loss_prior.item(),
                    'Betas/beta': kl_beta,
                    'Gammas/gamma_local' : gamma_local.item(),
                    'Gammas/gamma_global' : gamma_global.item(),
                    'Losses/average_tokens':average_tokens,
                    'KL/prior_variance_mean' : prior_variance_mean.item(),
                    "SR/sr_loss" : sr_loss.item(),
                    "SR/sr_lambda" : 10,      
                    }    
               
                wandb.log(log_dict)
                # ckpt_path = os.path.join(save_dir, f"checkpoint_ep{epoch}_step{batch_idx}.pt")
    #             torch.save(model.state_dict(), ckpt_path)
    #             print(f"Saved checkpoint: {ckpt_path}")
    #             checkpoints = sorted(glob.glob(os.path.join(save_dir, "checkpoint_*.pt")), key=os.path.getmtime)
    #             while len(checkpoints) > keep_last_n:
    #                 oldest_ckpt = checkpoints.pop(0)
    #                 os.remove(oldest_ckpt)
    # final_path = os.path.join(save_dir, "hvae_final_model.pt")
    # torch.save(model.state_dict(), final_path)
    # print(f"Training complete. Final model saved to {final_path}")

            
train()