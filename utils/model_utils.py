import importlib
import torch


# ----------General----------

def get_model(config):
    """
    Dynamically imports the model module and instantiates the class.
    Assumes model_name maps to a file in the 'models' directory.
    """
    model_name = config['model']
    try:
        module = importlib.import_module(f'models.{model_name}')
        model_class = getattr(module, 'MODEL_CLASS')
        return model_class(**config['model_config'], device=torch.device(config['device']))
        
    except ImportError as e:
        raise ImportError(f"Could not find model file for '{model_name}'. Ensure models/{model_name}.py exists.") from e
    except AttributeError as e:
        raise AttributeError(f"Module '{model_name}' does not define a 'MODEL_CLASS' attribute.") from e


# ----------Causal conv related----------

def dilation(layer_number):
    return 2 ** (layer_number % 10)

def get_receptive_field_size(kernel_size, num_layers):
    n_full_cycles = num_layers // 10
    remainder = num_layers % 10
    dilation_sum = n_full_cycles * (2**10 - 1) + (2**remainder - 1)
    dilated_receptive_field = 1 + (kernel_size - 1) * dilation_sum
    return (kernel_size - 1) + dilated_receptive_field

# ----------Transformer related----------

def causal_attn_mask(sz, device=torch.device('cpu')):
    """Generate a square causal mask for the sequence.
    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
