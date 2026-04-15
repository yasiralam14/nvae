import torch
import torch.nn as nn
from transformers import BartModel
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class GlobalToSoftTokens(nn.Module):
    def __init__(self, global_latent_dim, bart_hidden_dim, k=4):
        super().__init__()
        self.k = k
        self.bart_hidden_dim = bart_hidden_dim
        
        intermediate_dim = global_latent_dim * 2 
        
        self.mlp = nn.Sequential(
            nn.Linear(global_latent_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, k * bart_hidden_dim)
        )

    def forward(self, z_global):
        batch_size = z_global.size(0)
        flat_tokens = self.mlp(z_global)
        # Reshape to (batch_size, k, bart_hidden_dim)
        return flat_tokens.view(batch_size, self.k, self.bart_hidden_dim)

class ResidualShifter(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        last_layer = nn.Linear(hidden_dim, latent_dim * 2)
        nn.init.normal_(last_layer.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(last_layer.bias)
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.SiLU(),
            last_layer
        )
        
    def forward(self, x, base_mu, base_raw_var):
        delta = self.net(x)
        delta_mu, delta_raw_var = torch.chunk(delta, 2, dim=-1)
        
        new_mu = base_mu + delta_mu
        new_raw_var = base_raw_var + delta_raw_var
        
        sigma = F.softplus(new_raw_var) + 1e-8
        new_sigma2 = sigma.pow(2)
        
        return new_mu, new_sigma2

def reparameterize(mu, sigma2):
    std = torch.sqrt(sigma2)
    eps = torch.randn_like(std)
    return mu + eps * std

class BartHVAE(nn.Module):
    def __init__(self, hyperParams):
        super().__init__()
        model_name = hyperParams['model_name']
        local_latent_dim = hyperParams['local_latent_dim']
        global_latent_dim = hyperParams['global_latent_dim']
        self.mask_prob = hyperParams.get('mask_prob', 0)
        self.bart = BartModel.from_pretrained(model_name)
        self.bart_hidden_dim = self.bart.config.d_model
        self.global_shifter = ResidualShifter(self.bart_hidden_dim, self.bart_hidden_dim, global_latent_dim)
        local_prior_layer = nn.Linear(self.bart_hidden_dim, local_latent_dim * 2)
        nn.init.normal_(local_prior_layer.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(local_prior_layer.bias)
        self.local_prior_net = local_prior_layer
        self.local_shifter = ResidualShifter(self.bart_hidden_dim * 2, self.bart_hidden_dim, local_latent_dim)
        self.global_emb = GlobalToSoftTokens(global_latent_dim,self.bart_hidden_dim, k = 10)
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, self.bart_hidden_dim))
        self.latent_to_bart = nn.Linear(local_latent_dim, self.bart.config.d_model)
        self.latent_to_bart_mlp = nn.Sequential(
            nn.Linear(local_latent_dim, self.bart_hidden_dim),
            nn.GELU(),
            nn.Linear(self.bart_hidden_dim, self.bart_hidden_dim),
            nn.LayerNorm(self.bart_hidden_dim) # Stabilizes the vectors before the final dot product
        )
        self.latent_to_bart2 = nn.Linear(local_latent_dim, self.bart.config.d_model)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.bart.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        local_features = encoder_outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(local_features.size()).float()
        sum_features = (local_features * mask_expanded).sum(dim=1)
        seq_lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)
        global_feature = sum_features / seq_lengths
        
        batch_size = global_feature.size(0)
        device = global_feature.device
        
        base_g_mu = torch.zeros(batch_size, self.global_shifter.net[-1].out_features // 2, device=device)
        base_g_logvar = torch.zeros_like(base_g_mu)
        
        g_mu, g_sigma2 = self.global_shifter(global_feature, base_g_mu, base_g_logvar)
        z_global = reparameterize(g_mu, g_sigma2)
        
        z_global_expanded = z_global.unsqueeze(1)
        globa_emb = self.global_emb(z_global_expanded)
        decoder_outputs = self.bart.decoder(
            input_ids=input_ids,
            encoder_hidden_states=globa_emb,
            return_dict=True
        )
        top_down_features = decoder_outputs.last_hidden_state
        
        prior_params = self.local_prior_net(top_down_features)
        l_prior_mu, l_prior_raw_var = torch.chunk(prior_params, 2, dim=-1)
        l_prior_sigma = F.softplus(l_prior_raw_var) + 1e-8
        l_prior_sigma2 = l_prior_sigma.pow(2)
        
        masked_local_features = local_features
        if self.training and self.mask_prob > 0:
            seq_len = local_features.size(1)
            drop_mask = torch.rand(batch_size, seq_len, 1, device=device) < self.mask_prob
            masked_local_features = torch.where(drop_mask, self.mask_embedding.expand_as(local_features), local_features)
        
        combined_features = torch.cat([masked_local_features, top_down_features], dim=-1)
        l_post_mu, l_post_sigma2 = self.local_shifter(combined_features, l_prior_mu, l_prior_raw_var)

        z_local_prior = reparameterize(l_prior_mu, l_prior_sigma2)
        projected_z_prior = self.latent_to_bart_mlp(z_local_prior)
        logits_prior = F.linear(projected_z_prior, self.bart.shared.weight)
        z_local = reparameterize(l_post_mu, l_post_sigma2)
        batch_size, seq_len, _ = z_local.size()
        z_global_expanded_seq = z_global.unsqueeze(1).expand(batch_size, seq_len, -1)
        z_combined = torch.cat([z_local, z_global_expanded_seq], dim=-1)
        projected_z = self.latent_to_bart_mlp(z_local)
        logits = F.linear(projected_z, self.bart.shared.weight)
                
        return {
            "logits": logits,
            'logits_prior': logits_prior,
            "g_mu": g_mu, "g_sigma2": g_sigma2,
            "l_prior_mu": l_prior_mu, "l_prior_sigma2": l_prior_sigma2,
            "l_post_mu": l_post_mu, "l_post_sigma2": l_post_sigma2
        }
        
    @torch.no_grad()
    def reconstruct(self, input_ids, attention_mask):
        device = input_ids.device
        seq_len = input_ids.size(1)
        start_token_id = self.bart.config.decoder_start_token_id or self.bart.config.bos_token_id

        encoder_outputs = self.bart.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        local_features = encoder_outputs.last_hidden_state
        
        mask_expanded = attention_mask.unsqueeze(-1).float()
        global_feature = (local_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        base_g_mu = torch.zeros(1, self.global_shifter.net[-1].out_features // 2, device=device)
        g_mu, g_sigma2 = self.global_shifter(global_feature, base_g_mu, torch.zeros_like(base_g_mu))
        z_global = reparameterize(g_mu, g_sigma2)
        globa_emb = self.global_emb(z_global.unsqueeze(1))
        
        generated_ids = torch.tensor([[start_token_id]], device=device)
        
        for t in range(seq_len):
            decoder_outputs = self.bart.decoder(
                input_ids=generated_ids,
                encoder_hidden_states=globa_emb,
                return_dict=True
            )
            top_down_features = decoder_outputs.last_hidden_state[:, -1:, :]
            
            prior_params = self.local_prior_net(top_down_features)
            l_prior_mu, l_prior_raw_var = torch.chunk(prior_params, 2, dim=-1)
            
            current_local_feature = local_features[:, t:t+1, :]
            combined_features = torch.cat([current_local_feature, top_down_features], dim=-1)
            l_post_mu, l_post_sigma2 = self.local_shifter(combined_features, l_prior_mu, l_prior_raw_var)
            
            z_local = reparameterize(l_post_mu, l_post_sigma2)
            
            logits = self.lm_head(z_local)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if next_token.item() == self.bart.config.eos_token_id:
                break
                
        return generated_ids
    @torch.no_grad()
    def generate(self, max_length=50, device='cpu'):
        start_token_id = self.bart.config.decoder_start_token_id or self.bart.config.bos_token_id
        
        global_latent_dim = self.global_emb.in_features
        z_global = torch.randn(1, global_latent_dim, device=device)
        global_emb = self.global_emb(z_global.unsqueeze(1))
        
        generated_ids = torch.tensor([[start_token_id]], device=device)
        
        for _ in range(max_length):
            decoder_outputs = self.bart.decoder(
                input_ids=generated_ids,
                encoder_hidden_states=global_emb,
                return_dict=True
            )
            top_down_features = decoder_outputs.last_hidden_state[:, -1:, :]
            
            prior_params = self.local_prior_net(top_down_features)
            l_prior_mu, l_prior_raw_var = torch.chunk(prior_params, 2, dim=-1)
            l_prior_sigma2 = (F.softplus(l_prior_raw_var) + 1e-8).pow(2)
            
            z_local = reparameterize(l_prior_mu, l_prior_sigma2)
            
            logits = self.lm_head(z_local)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if next_token.item() == self.bart.config.eos_token_id:
                break
                
        return generated_ids