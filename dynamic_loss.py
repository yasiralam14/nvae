import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F


class Nvae_Loss_Dynamic(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.pad_idx = hyperparameters["bart_pad_id"]
        self.local_latent_dim = hyperparameters["local_latent_dim"]

        self.cross_entropy_loss = nn.CrossEntropyLoss(
            ignore_index=self.pad_idx, reduction="none"
        )

        self.free_bits_global = 0.5
        self.free_bits_local = 0.4

    def forward(
    self,
    mu_t_q, sigma2_t_q,
    mu_i_q, sigma2_i_q,
    mu_i_p, sigma2_i_p,
    reconstruction_logits,
    prior_logits,
    target_ids, word_mask,
    local_kl_beta=0.5, global_kl_beta=0.1,
    is_warmup=True,
    sr_loss=0.0, sr_lambda=0.01
    ):
        eps = 1e-9
        B, W = target_ids.shape
        
        
        shift_logits = reconstruction_logits[..., :-1, :].contiguous()
        shift_logits_prior = prior_logits[..., :-1, :].contiguous()
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

        per_token_loss_prior = F.cross_entropy(
            shift_logits_prior.view(-1, reconstruction_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=self.pad_idx,
            reduction='none'
        )
        
        per_token_loss_prior = per_token_loss_prior.view(B, -1)
        
        reconstruction_loss_prior = per_token_loss_prior.sum(dim=1).mean()

       
        shift_mask = (shift_labels != self.pad_idx).float()
        num_active_tokens = shift_mask.sum()
        mean_token_loss_prior = per_token_loss_prior.sum() / (num_active_tokens + eps)


        
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
        
        mask_float1 = word_mask.float().unsqueeze(-1)
        
        masked_sigma2 = sigma2_i_p * mask_float1
        
        prior_variance_norm = torch.norm(masked_sigma2)
        
        valid_elements = mask_float1.sum() * self.local_latent_dim
        prior_variance_mean = masked_sigma2.sum() / (valid_elements + eps)
        
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
        
        if is_warmup:
            s_global = 1.0 
            s_local = (num_active_tokens / B).clamp(min=1.0) 
            
            W_global = s_global * global_kl_raw_mean.detach()
            W_local = s_local * local_kl_raw_mean.detach()
            
            total_W = W_global + W_local + eps
            
            gamma_global = (W_global / total_W) * 1.0
            gamma_local = (W_local / total_W) * 1.0
        else:
            if global_kl_beta > 0 or local_kl_beta > 0:
                # 1. Base relative balancing using distance from 1.0
                active_global = (1.0 - global_frac_dims_clamped)
                active_local = (1.0 - local_frac_dims_clamped)
                
                alpha = 6.0 

                active_global_adj = torch.pow(active_global, alpha).clamp(min=eps).detach()
                active_local_adj = torch.pow(active_local, alpha).clamp(min=eps).detach()

                total_active_adj = active_global_adj + active_local_adj

                gamma_global = (active_global_adj / total_active_adj) * 2.0
                gamma_local = (active_local_adj / total_active_adj) * 2.0
                
                # 2. Smooth penalty for exceeding 0.9
                # (1.0 - clamp) / 0.1 maps a clamp of 0.9 to 1.0 (no penalty) 
                # and a clamp of 0.99 to 0.1 (severe penalty)
                base_penalty_global = ((1.0 - global_frac_dims_clamped) / 0.3).clamp(min=eps, max=1.0).detach()
                base_penalty_local = ((1.0 - local_frac_dims_clamped) / 0.3).clamp(min=eps, max=1.0).detach()

                aggressiveness_factor = 6.0 
                penalty_global = torch.pow(base_penalty_global, aggressiveness_factor).clamp(min=eps, max=1.0).detach()
                penalty_local = torch.pow(base_penalty_local, aggressiveness_factor).clamp(min=eps, max=1.0).detach()
                # Apply the penalty
                gamma_global = gamma_global * penalty_global
                gamma_local = gamma_local * penalty_local
            else:
                gamma_global = torch.tensor(1.0, device=target_ids.device)
                gamma_local = torch.tensor(1.0, device=target_ids.device)

        # gamma_global = torch.tensor(1.0)
        # gamma_local = torch.tensor(1.0)
        weighted_global_kl = global_kl_beta * gamma_global * global_kl_loss
        weighted_local_kl = local_kl_beta * gamma_local * local_kl_loss
        
        total_kl = weighted_global_kl + weighted_local_kl
        total_loss = reconstruction_loss + total_kl + 2*reconstruction_loss_prior  # + sr_lambda * sr_loss

        kl_ratio = (reconstruction_loss / (total_kl + eps)).detach()
        average_tokens = num_active_tokens/B
        
        loss_dict = {
            'total_loss': total_loss,
            'reconstruction_loss' : reconstruction_loss,
            'global_kl_loss' : global_kl_loss,
            'local_kl_loss' : local_kl_loss,
            'gamma_global': gamma_global,
            'gamma_local': gamma_local,
            'kl_ratio' : kl_ratio,
            'mean_token_loss' : mean_token_loss,
            'mean_token_loss_prior':mean_token_loss_prior,
            'global_kl_raw_mean' : global_kl_raw_mean,
            'local_kl_raw_mean' : local_kl_raw_mean,
            'global_frac_dims_clamped' : global_frac_dims_clamped,
            'local_frac_dims_clamped' : local_frac_dims_clamped,
            'average_tokens': average_tokens,
            'prior_variance_mean':prior_variance_mean
        }

        return loss_dict

