"""
This script is used to inspect a checkpoint file and print the statistics for the model state, EMA state, and optimizer state.
"""

import torch
import argparse
import os

def inspect_checkpoint(checkpoint_path):
    """
    Loads a checkpoint and prints statistics for both the model state and EMA state.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    print(f"Inspecting checkpoint: {checkpoint_path}\n")

    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # --- Inspect model_state_dict ---
    if 'model_state_dict' in checkpoint:
        print("--- Model State (model_state_dict) ---")
        model_state = checkpoint['model_state_dict']
        print_state_stats(model_state)
    else:
        print("--- 'model_state_dict' not found in checkpoint ---")

    # --- Inspect ema_state_dict ---
    if 'ema_state_dict' in checkpoint:
        print("\n--- EMA State (ema_state_dict) ---")
        ema_state = checkpoint['ema_state_dict']
        print_state_stats(ema_state)
    else:
        print("\n--- 'ema_state_dict' not found in checkpoint ---")
        
    if 'optimizer_state_dict' in checkpoint:
        print("\n--- Optimizer State (optimizer_state_dict) ---")
        optimizer_state = checkpoint['optimizer_state_dict']
        print_optimizer_stats(optimizer_state)
    else:
        print("\n--- 'optimizer_state_dict' not found in checkpoint ---")
        
    if 'epoch' in checkpoint:
        print(f"\nEpoch: {checkpoint['epoch']}")
    else:
        print("\n--- 'epoch' not found in checkpoint ---")


def print_state_stats(state_dict):
    """
    Prints the norm, sum, and mean for a few key layers in a state_dict.
    """
    # We'll inspect a few representative layers
    layers_to_inspect = {
        'input_projection.weight': 'Input Projection Weights',
        'seqTransDecoder.stack.0.self_attn.in_proj_weight': 'Decoder Layer 0 Self-Attention Weights',
        'final_layer.bias': 'Final Layer Bias'
    }

    for layer_name, description in layers_to_inspect.items():
        if layer_name in state_dict:
            weights = state_dict[layer_name]
            norm = torch.linalg.norm(weights).item()
            sum_val = torch.sum(weights).item()
            mean_val = torch.mean(weights).item()
            print(f"  - {description} ({layer_name}):")
            print(f"    - Norm: {norm:.6f}")
            print(f"    - Sum:  {sum_val:.6f}")
            print(f"    - Mean: {mean_val:.6f}")
        else:
            print(f"  - Layer '{layer_name}' not found in this state dict.")


def print_optimizer_stats(optimizer_state):
    """
    Prints hyperparameters and statistics for optimizer state buffers.
    """
    if not optimizer_state:
        return

    # Print main hyperparameters from the first parameter group
    if 'param_groups' in optimizer_state and optimizer_state['param_groups']:
        param_group = optimizer_state['param_groups'][0]
        print(f"  - Learning Rate: {param_group['lr']}")
        print(f"  - Weight Decay: {param_group['weight_decay']}")
    
    # Print statistics for the internal state buffers
    if 'state' in optimizer_state:
        print("  - Optimizer State Buffers:")
        state_buffers = optimizer_state['state']
        # Sort by param_id for consistent ordering
        sorted_param_ids = sorted(state_buffers.keys())
        
        # Print stats for the first few parameter buffers as a sample
        for i, param_id in enumerate(sorted_param_ids[:3]):
            print(f"    - Parameter ID Group {param_id}:")
            buffers = state_buffers[param_id]
            for buffer_name, tensor in buffers.items():
                if isinstance(tensor, torch.Tensor):
                    norm = torch.linalg.norm(tensor).item()
                    sum_val = torch.sum(tensor).item()
                    print(f"      - Buffer '{buffer_name}': Norm={norm:.6f}, Sum={sum_val:.6f}")
        if len(sorted_param_ids) > 3:
            print("      ...")

    else:
        print("  - No 'state' buffers found in optimizer state dict.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect a PyTorch checkpoint file.')
    parser.add_argument('checkpoint_path', type=str, help='Path to the .pt checkpoint file.')
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint_path)
