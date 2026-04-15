

def get_kl_beta(
    epoch_index,
    batch_idx,
    steps_per_epoch,
    MIN_BETA=0.001,  
    MAX_BETA=0.1,    
    CYCLE_EPOCHS=20, 
    MIN_HOLD_FRAC=0.2, 
    RAMP_FRAC=0.5      
):
    """
    Cyclical Beta Schedule:
    [MIN_BETA ..... / RAMP / ..... MAX_BETA .....] -> Repeat
    """
    total_steps_in_cycle = steps_per_epoch * CYCLE_EPOCHS
    current_global_step = epoch_index * steps_per_epoch + batch_idx

    cycle_progress = (current_global_step % total_steps_in_cycle) / total_steps_in_cycle

    phase_a_end = MIN_HOLD_FRAC
    phase_b_end = MIN_HOLD_FRAC + RAMP_FRAC

    if cycle_progress < phase_a_end:
        current_beta = MIN_BETA

    elif cycle_progress < phase_b_end:
        ramp_progress = (cycle_progress - phase_a_end) / RAMP_FRAC
        current_beta = MIN_BETA + ramp_progress * (MAX_BETA - MIN_BETA)

    else:
        current_beta = MAX_BETA

    return (current_beta, current_beta)