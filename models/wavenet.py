import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import mu_law_quantize, dequantize, inverse_mu_law
from models.causal_conv import CausalConvNet
from models.base import BaseModel


class EEGPredictor(BaseModel):

    def _build_model(self):
        return Wavenet(**self.config)


class Wavenet(nn.Module):
    def __init__(self,
                 n_channels,
                 kernel_size,
                 n_blocks,
                 device=torch.device('cpu')):

        super(Wavenet, self).__init__()

        self.device = device

        self.cc = CausalConvNet(n_channels,
                                kernel_size,
                                n_blocks,
                                out_channels=128,
                                device=device).to(device)
        
        self.out = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)

    def forward(self,
                x,
                len_y=1,
                sampling=True,
                n_samples=1):
        """
        Params:
        x: Inputs of shape (batch, 1, len_x). Values are expected to be [-100, 100]
        len_y: Used during eval mode. Tells how long the generated forecast should be
        sampling: Used during eval mode. If set to True, next time step is sampled from the
                output distribution, otherwise the index with the highest probability is drawn
        n_samples: Used during eval mode. Indicates how many samples per input should be drawn

        return: Train mode: output logits over the next time step, shape (batch, 256, len_x-rf+1)
                Eval mode: predicted next len_y time steps, shape (batch*n_samples, len_y)
        """

        # Quantize & dequantize x
        x = x / 100  # normalize to [-1, 1]
        x = mu_law_quantize(x)
        x = dequantize(x)

        if self.training:

            z = self.cc(x)
            out = self.out(z)

            return out
        
        else:
            if not sampling and n_samples > 1:
                warnings.warn("n_samples > 1 with sampling=False produces identical predictions. " \
                "Set sampling=True or n_samples=1 to avoid unnecessary computation.")

            rf = self.cc.receptive_field_size

            for i in range(len_y):
                # Get logits
                z = self.cc(x[:, :, -rf:])
                logits = self.out(z)  # (n, 256, 1)

                # Sample
                probs = F.softmax(logits[:, :, -1], dim=1)   # (n, 256)

                if i==0 and n_samples > 1:
                    x = x.repeat_interleave(n_samples, dim=0)
                    probs = probs.repeat_interleave(n_samples, dim=0)

                if sampling:
                    idx = torch.multinomial(probs, num_samples=1)   # (n, 1)
                else:
                    idx = torch.argmax(probs, dim=1, keepdim=True)  # (n, 1)

                # Dequantize and concat to input 
                x_pred = dequantize(idx)
                x = torch.cat([x, x_pred.unsqueeze(1)], dim=2)
            
            # Inverse mu law and normalization
            y = x[:, :, -len_y:].view(-1, len_y)
            y = inverse_mu_law(y)
            y *= 100  # back to [-100, 100]

            return y
