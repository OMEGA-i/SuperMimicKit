"""Small, unquantized policy checkpoints for reproducible evaluation."""
from pathlib import Path
import os
import torch


def save_evaluation_checkpoint(checkpoint, output):
    output = Path(output)
    if output.exists():
        return False
    saved = {k: checkpoint[k] for k in (
        'config', 'joint_names', 'body_names', 'args', 'iteration',
        'learner_version', 'training_contract',
    )}
    # Clone CPU tensors so saving never includes storage from the replay buffer.
    # Actor state_dict includes the observation/goal normalization buffers.
    saved['actor'] = {k: v.detach().cpu().clone() for k, v in checkpoint['actor'].items()}
    temp = output.with_name(output.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(saved, temp)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    return True
