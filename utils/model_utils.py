import os
import json
import importlib
import torch


# ----------General----------

RVQ_MODELS = {'revbrain', 'tf_revbrain'}

MODEL_CONFIGS = {
    'revbrain': {
        'embedding_dim': 8,
        'enc_channels': 32,
        'enc_kernel': 2,
        'enc_blocks': 10,
        'z_dim': 256,
        'mlp_hidden_dim': 512,
    },
    'tf_revbrain': {
        'enc_channels': 32,
        'enc_kernel': 2,
        'enc_blocks': 10,
        'd_model': 256,
        'n_head': 4,
        'n_transformer_layers': 4,
        'd_transformer_ff': 1024,
    },
    'wavenet': {
        'n_channels': 32,
        'kernel_size': 2,
        'n_blocks': 10,
    },
}

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
    
def load_run(run_dir, device):
    with open(os.path.join(run_dir, 'training_config.json')) as f:
        config = json.load(f)
    config['device'] = device
    model = get_model(config)
    model.load_state_dict(
        torch.load(os.path.join(run_dir, 'model.pth'),
                   map_location=torch.device(device))
    )
    return model, config


# ----------Inference----------

def run_rvq_inference(model, x, codebooks, n_samples, sampling, in_relat_enc):
    """
    Params:
    x: input to model, shape (batch, 1, len)
    codebooks: long tensor containing codebooks, shape (M, N, len_y)
    n_samples: how many samples to create per input
    
    return: predictions of shape (batch, n_samples, len_y)
    """

    # model inference
    indxs = model(x, n_samples=n_samples, sampling=sampling)  # (M, n_samples*batch)

    # get codebook entries based on predicted indices
    indxs = indxs.unsqueeze(-1).expand(-1, -1, codebooks.shape[2])
    predictions = torch.gather(codebooks, dim=1, index=indxs)  # (M, batch*n_samples, len_y)

    # sum cb layers and add back input relative encoding value
    predictions = torch.sum(predictions, dim=0)  # (batch*n_samples, len_y)
    predictions = predictions.view(-1, n_samples, predictions.shape[1])
    if in_relat_enc:
        predictions = predictions + x[:, :, -1:]

    return predictions


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
