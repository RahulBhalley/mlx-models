# MLX-Models
Provides various pre-trained models for Apple's MLX framework.

## Installation

Open your terminal and run the following command:

```bash
pip install "git+https://github.com/RahulBhalley/mlx-models.git"
```

This command will fetch the latest version of the MLX-Models package from the GitHub repository and install it on your system.

## Usage

### Loading Pre-trained Models

MLX-Models simplifies the process of loading and utilizing pre-trained models. For instance, to load a pre-trained VGG19 model, you can use the following code snippet:

```python
import mlx.models as mlx_models

# Load the pre-trained VGG19 model
vgg19 = mlx_models.vision.VGG19(load_weights=True)
```

This will initialize the VGG19 model with pre-trained weights.
å
### Available Models

At present, the MLX-Models package primarily supports vision models. Later, we'll add support for audio and text models as well.

## Contributing

We welcome contributions to the MLX-Models project! Whether it's adding new models, improving the existing ones, or fixing bugs, your contributions are valuable to us. Just create a pull request (PR).

## License

This project is licensed under the MIT License.

## Support

If you encounter any issues please file an issue and if you have questions please use Discussions.