"""Small, unquantized network checkpoints for evaluation and demo export."""
from pathlib import Path
import os
import torch


def save_evaluation_checkpoint(checkpoint, output, include_world=False):
    output = Path(output)
    if output.exists():
        return False
    saved = {k: checkpoint[k] for k in (
        'config', 'joint_names', 'body_names', 'args', 'iteration',
        'learner_version', 'training_contract',
    )}
    # Clone CPU tensors so saving never includes storage from the replay buffer.
    # Actor state_dict includes the observation/goal normalization buffers.
    for name in ('actor', 'world') if include_world else ('actor',):
        saved[name] = {k: v.detach().cpu().clone() for k, v in checkpoint[name].items()}
    temp = output.with_name(output.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(saved, temp)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    return True
