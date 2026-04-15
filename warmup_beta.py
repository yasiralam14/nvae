def get_kl_beta_and_warmup_flag(epoch_idx, batch_idx, steps_per_epoch, warmup_epochs, max_beta=1.0):
    current_step = (epoch_idx * steps_per_epoch) + batch_idx
    total_warmup_steps = warmup_epochs * steps_per_epoch
    
    if current_step >= total_warmup_steps or total_warmup_steps == 0:
        return max_beta, False
        
    current_beta = max_beta * (current_step / total_warmup_steps)
    return current_beta, True