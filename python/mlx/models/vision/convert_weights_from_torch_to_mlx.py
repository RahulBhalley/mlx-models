# Copyright ©️ 2024 Rahul Bhalley

import torch
import mlx.core as mx
from torchvision import models

import os
import re
import json
import argparse

__all__ = []


def convert_torch_tensor_to_mx_array(torch_tensor: torch.Tensor) -> mx.array:
    """
    Convert a PyTorch tensor to an MLX array. If the tensor is 4-dimensional (e.g., from a convolutional layer),
    it will be converted to channel-last format. Otherwise, it will be converted as is.

    Parameters:
    - torch_tensor: A PyTorch Tensor.

    Returns:
    - An MLX array, with 4D tensors converted to channel-last format.
    """
    if torch_tensor.dim() == 4:
        mx_array = torch_tensor.permute(0, 2, 3, 1).data.numpy()
    else:
        mx_array = torch_tensor.data.numpy()
    return mx.array(mx_array)


def replace_string(input_string: str) -> str:
    """
    Replace specific patterns in a string to conform to MLX naming conventions.

    Parameters:
    - input_string: The original string.

    Returns:
    - The modified string with replaced patterns.
    """
    pattern = r'(\b\w+\.)+(\d+\.)(weight|bias)'
    def repl(match):
        return f'{match.group(1)}layers.{match.group(2)}{match.group(3)}'
    replaced_string = re.sub(pattern, repl, input_string)
    return replaced_string


def convert_weights_from_torch_to_mlx(torch_filepath: str, model_name: str, format: str = "safetensors") -> None:
    """
    Convert PyTorch model weights to MLX format.

    Parameters:
    - torch_filepath: Path to the PyTorch weights file.
    - format: The target format for the weights. Default is "safetensors".
    """
    if format not in ["safetensors", "npz"]:
        print(f'Format argument is {format}. It must be either "safetensors" or "npz".')
        return

    try:
        torch_weights = torch.load(torch_filepath)
    except Exception as e:
        print(f"Unable to read {torch_filepath}: {e}")
        return
    
    mlx_weights = {}
    for torch_key, torch_value in torch_weights.items():
        mlx_key = replace_string(torch_key)
        mlx_value = convert_torch_tensor_to_mx_array(torch_value)
        mlx_weights[mlx_key] = mlx_value

    # Define the directory to save MLX models
    mlx_cache_dir = os.path.expanduser('~/.cache/mlx.models/checkpoints')
    os.makedirs(mlx_cache_dir, exist_ok=True)  # Ensure the directory exists

    mlx_filename = os.path.join(mlx_cache_dir, f"{os.path.basename(torch_filepath).rsplit('.', 1)[0]}.{format}")
    if format == "safetensors":
        mx.save_safetensors(mlx_filename, mlx_weights)
    else:
        mx.savez(mlx_filename, **mlx_weights)

    # After saving the MLX model, update the model mappings
    update_model_mappings(model_name, mlx_filename)


def load_and_cache_model(model_name):
    """
    Load and cache a PyTorch model using torchvision.

    Parameters:
    - model_name: Name of the model to load.

    Returns:
    - Path to the cached model file.
    """
    model = getattr(models, model_name)(weights=True)

    cache_dir = os.path.expanduser(os.getenv('TORCH_HOME', '~/.cache/torch'))
    model_cache_dir = os.path.join(cache_dir, 'hub', 'checkpoints')

    # model_pattern = re.compile(rf'{model_name}-[a-f0-9]+\.pth')
    model_pattern = re.compile(rf'\b{model_name}-.*(\.pth|\.pt)\b')
    
    for filename in os.listdir(model_cache_dir):
        if model_pattern.match(filename):
            return os.path.join(model_cache_dir, filename)

    raise FileNotFoundError(f"{model_name} model file not found in cache.")


def update_model_mappings(model_name, mlx_filename):
    # Define the path for the JSON file that will store model mappings
    mappings_file = os.path.expanduser('~/.cache/mlx.models/model_mappings.json')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(mappings_file), exist_ok=True)

    # Load existing mappings if the file exists, otherwise initialize an empty dict
    if os.path.exists(mappings_file):
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
    else:
        mappings = {}

    # Update the mappings with the new model
    mappings[model_name] = mlx_filename

    # Save the updated mappings back to the JSON file
    with open(mappings_file, 'w') as f:
        json.dump(mappings, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to MLX format.")
    parser.add_argument("model_name", type=str, help="Name of the PyTorch model to convert (e.g., 'vgg19').")
    parser.add_argument("--format", type=str, default="safetensors", choices=["safetensors", "npz"], help="Target format for the weights.")

    args = parser.parse_args()

    torch_model_path = load_and_cache_model(args.model_name)
    convert_weights_from_torch_to_mlx(torch_model_path, args.model_name, args.format)
