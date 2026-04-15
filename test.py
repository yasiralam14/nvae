import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from transformers import BartTokenizer
from .model import BartHVAE
from .loss import Nvae_Loss
from .get_kl_beta import get_kl_beta
import wandb
import pandas as pd
import os
import glob



wandb.login(key="0ce56922c7ea30310a87d49246b15bc7d7ca9c89")

hyperParams = {}
wandb.init(
    project="nvae_text",
    name = 'mean_pooling_no_dropout'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
df = pd.read_parquet("/home/salam4/hvae_project/word_hvae/ht_hvae/HVAE/data/sentences_df.parquet")
def strip_bos_eos(text: str) -> str:
    text = text.replace("<BOS>", "").replace("<EOS>", "")
    text = text.replace("</s>", " [EOS] ").replace("<s>", " [BOS] ")
    return " ".join(text.split()).strip()

df['clean_text'] = df['sentence'].apply(strip_bos_eos)
encodings = tokenizer(
    df['clean_text'].tolist(),
    max_length=50,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)

input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

dataset = TensorDataset(input_ids, attention_mask)
    
train_dataset, val_dataset = train_test_split(dataset, test_size=0.01, random_state=42)

train_loader = DataLoader(
    train_dataset, 
    batch_size=256, 
    shuffle=True,       
    num_workers=4,      
    pin_memory=True     
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=64, 
    shuffle=False       
)


print(f"BOS Token ID (<s>):    {tokenizer.bos_token_id}")
print(f"PAD Token ID (<pad>):  {tokenizer.pad_token_id}")
print(f"EOS Token ID (</s>):   {tokenizer.eos_token_id}")

hyperParams['bart_pad_id'] = tokenizer.pad_token_id
hyperParams['local_latent_dim'] = 32
hyperParams['global_latent_dim'] = 64
hyperParams['model_name'] = 'facebook/bart-base'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartHVAE(hyperParams).to(device)

ckpt_path = '/home/salam4/hvae_project/nvae/hvae_checkpoints/hvae_final_model.pt'
    
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

model.eval()


print("Testing Training Reconstruction")
train_batch = next(iter(train_loader))
train_input_ids = train_batch[0][0].unsqueeze(0).to(device)
train_attention_mask = train_batch[1][0].unsqueeze(0).to(device)
train_recon_ids = model.reconstruct(train_input_ids, train_attention_mask)
train_real_text = tokenizer.decode(train_input_ids[0], skip_special_tokens=True)
train_recon_text = tokenizer.decode(train_recon_ids[0], skip_special_tokens=True)
print(f"Train Real Text: {train_real_text}")
print(f"Train Recon Text: {train_recon_text}")


print("Testing Validation Reconstruction")
val_batch = next(iter(val_loader))
val_input_ids = val_batch[0][0].unsqueeze(0).to(device)
val_attention_mask = train_batch[1][0].unsqueeze(0).to(device)
val_recon_ids = model.reconstruct(val_input_ids, val_attention_mask)

val_real_text = tokenizer.decode(val_input_ids[0], skip_special_tokens=True)
val_recon_text = tokenizer.decode(val_recon_ids[0], skip_special_tokens=True)
print(f"Val Real Text: {val_real_text}")
print(f"Val Recon Text: {val_recon_text}")


print("Testing Free hand generation")
generated_ids = model.generate(device = device)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text}")




