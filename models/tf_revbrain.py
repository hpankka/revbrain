import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from utils.model_utils import causal_attn_mask
from models.causal_conv import CausalConvNet


class TfREVBrain(nn.Module):

    def __init__(self,
                 M,                     # Number of codebook levels
                 N,                     # Codebook size
                 enc_channels,          # Intermediate channels in Wavenet-encoder
                 enc_kernel,            # Encoder kernel size
                 enc_blocks,            # Number of Wavenet blocks in the encoder
                 d_model,               # Model dimension
                 n_head,                # Number of Transformer heads
                 n_transformer_layers,  # Number of Transformer layers
                 d_transformer_ff,      # Dimension of Transformer feedforward
                 device=torch.device('cpu')):

        super(TfREVBrain, self).__init__()

        self.device = device
        self.M = M
        self.N = N
        self.d_model = d_model

        self.encoder = CausalConvNet(n_channels=enc_channels,
                                     kernel_size=enc_kernel,
                                     n_blocks=enc_blocks,
                                     out_channels=d_model,
                                     device=device).to(device)
        self.receptive_field_size = self.encoder.receptive_field_size

        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=N,
                                                      embedding_dim=d_model)
                                        for m in range(M-1)])
        
        transformer_layer = TransformerEncoderLayer(d_model=d_model,
                                                    nhead=n_head,
                                                    dim_feedforward=d_transformer_ff,
                                                    batch_first=True)
        self.transformer = TransformerEncoder(transformer_layer,
                                              num_layers=n_transformer_layers)

        self.linears = nn.ModuleList([nn.Linear(in_features=d_model, out_features=N) for m in range(M)])

    def forward(self,
                x,
                targets=None,
                tf_with_eval=False,
                sampling=True,
                n_samples=1):
        """
        Train mode: forward pass runs with teacher forcing, i.e. ground truth targets are passed to
            the head. The output contains the output logits of each MLP block.
        Eval mode: predicted indices are passed to the head.
            The output is a LongTensor containing the predicted codebook indices of each MLP block.

        Params:
        x: Inputs of shape (batch, 1, len_x)
        targets: Targets for teacher forcing during training, shape (batch, M*len_train_target)
                For correct train behavior, make sure len_train_target=len_x-receptive_field_size+1
        tf_with_eval: set to True, when eval-mode should run with teacher forcing (e.g. validation step)
        sampling: Used during eval mode. If set to True, indices are sampled from the output distribution,
                otherwise the index with the highest probability is drawn
        n_samples: Used during eval mode. Indicates how many samples per input should be drawn

        return: Train mode: output logits of each MLP block, shape (batch, N, M*len_train_target)
                Eval mode: predicted codebook indexes of each MLP block, shape (batch*n_samples*M)
        """

        batch_size = x.shape[0]

        if self.training or tf_with_eval:
            assert targets is not None, 'Targets must be provided in train-mode'

            # Encoder
            z = self.encoder(x)  # (n, d_model, len_target)

            # Reshape
            z = z.permute(0, 2, 1).reshape(-1, 1, self.d_model)  # (n*len_target, 1, d_model)
            targets = targets.view(batch_size, self.M, -1)  # (n, M, len_target)
            targets = targets.permute(0, 2, 1).reshape(-1, self.M)  # (n*len_target, M)

            # Embed targets and concat to z
            for m, embedding in enumerate(self.embeddings):
                y = embedding(targets[:, m:m+1])  # (n*len_target, 1, d_model)
                z = torch.cat((z, y), dim=1)  # (n*len_target, M, d_model)

            # Head (Transformer)
            mask = causal_attn_mask(self.M, device=self.device)
            z = self.transformer(z, mask=mask, is_causal=True)  # (n*len_target, M, d_model)

            # Final linears
            out = torch.empty(z.shape[0], 0, self.N).to(self.device)
            for i, fc in enumerate(self.linears):
                fc_out = fc(z[:, i:i+1])  # (n*len_target, 1, N)
                out = torch.cat((out, fc_out), dim=1)  # (n*len_target, M, N)

            # Reshape
            out = out.view(batch_size, -1, self.M, self.N).permute(0, 3, 2, 1)  # (n, N, M, len_target)
            out = out.reshape(batch_size, self.N, -1)  # (n, N, M*len_target)

            return out

        else:
            batch_size *= n_samples

            # Encoder
            z = self.encoder(x[:, :, -self.receptive_field_size:])  # (n, d_model, 1)
            z = z.permute(0, 2, 1)  # (n, 1, d_model)

            # Head
            out = torch.empty(0, 1, dtype=torch.long).to(self.device)
            for m in range(self.M):
                if m > 0:
                    # Embed and concatenate predicted tokens
                    s = (m-1)*batch_size
                    y = self.embeddings[m-1](out[s:s+batch_size])  # (n, 1, d_model)
                    z = torch.cat((z, y), dim=1)

                # Get logits
                mask=causal_attn_mask(m+1, device=self.device)
                out_tf = self.transformer(z, mask=mask, is_causal=True)  # (n, 1, d_model)
                out_tf = self.linears[m](out_tf[:, m])  # (n, cb_size)

                # Sample
                out_tf = F.softmax(out_tf, dim=1)

                if m == 0 and n_samples > 1:
                    z = z.repeat_interleave(n_samples, dim=0)
                    out_tf = out_tf.repeat_interleave(n_samples, dim=0)
                    
                if sampling:
                    cur_indices = torch.multinomial(out_tf, num_samples=1)  # torch.int64, (batch_size, 1)
                else:
                    cur_indices = torch.argmax(out_tf, dim=1, keepdim=True)
                
                # Gather outputs
                out = torch.cat((out, cur_indices), dim=0)

            return torch.squeeze(out, dim=1)


MODEL_CLASS = TfREVBrain
