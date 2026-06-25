import torch
import torch.nn as nn
import torch.nn.functional as F

from models.causal_conv import CausalConvNet


class REVBrain(nn.Module):
    def __init__(self,
                 M,                 # Number of codebook levels
                 N,                 # Codebook size
                 enc_channels,      # Intermediate channels in Wavenet-encoder
                 enc_kernel,        # Encoder kernel size
                 enc_blocks,        # Number of Wavenet blocks in the encoder
                 z_dim,             # Dimension of the input representation
                 embedding_dim,     # Codebook index embedding size
                 mlp_hidden_dim,    # Dimension of MLP hidden
                 device=torch.device('cpu')):

        super(REVBrain, self).__init__()

        self.device = device
        self.M = M
        self.N = N
        
        self.encoder = CausalConvNet(n_channels=enc_channels,
                                     kernel_size=enc_kernel,
                                     n_blocks=enc_blocks,
                                     out_channels=z_dim,
                                     device=device).to(device)
        self.receptive_field_size = self.encoder.receptive_field_size

        self.mlp_blocks = nn.ModuleList([MlpBlock(N,
                                                  embedding_dim,
                                                  z_dim,
                                                  mlp_hidden_dim,
                                                  quantization_level=m).to(device) for m in range(M)])

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

            targets = targets.view(batch_size, self.M, -1)
            
            # Encoder
            z = self.encoder(x)

            # Head
            embedded_ys = torch.empty(batch_size, 0, targets.shape[2]).to(self.device)
            out = torch.empty(batch_size, self.N, 0).to(self.device)
            for m, mlp_ in enumerate(self.mlp_blocks):
                out_mlp, embedded_ys = mlp_(z, targets[:, m-1], embedded_ys)
                out = torch.cat((out, out_mlp), dim=2)

            return out
        else:
            batch_size *= n_samples

            z = self.encoder(x[:, :, -self.receptive_field_size:])  # (n, N, 1)

            embedded_ys = torch.empty(batch_size, 0, 1).to(self.device)
            out = torch.empty(0, 1, dtype=torch.long).to(self.device)
            cur_indices = torch.empty(batch_size, 1, dtype=torch.long).to(self.device)
            for m, mlp_ in enumerate(self.mlp_blocks):
                # Get logits
                out_mlp, embedded_ys = mlp_(z, cur_indices, embedded_ys)

                # Sample
                out_mlp = F.softmax(out_mlp[:, :, -1], dim=1)  # (n, N)

                if m==0 and n_samples > 1:
                    z = z.repeat_interleave(n_samples, dim=0)
                    out_mlp = out_mlp.repeat_interleave(n_samples, dim=0)
                    
                if sampling:
                    cur_indices = torch.multinomial(out_mlp, num_samples=1)  # torch.int64, (n, 1)
                else:
                    cur_indices = torch.argmax(out_mlp, dim=1, keepdim=True)
                
                # Gather outputs
                out = torch.cat((out, cur_indices), dim=0)
            
            return torch.squeeze(out, dim=1)


class MlpBlock(nn.Module):
    def __init__(self,
                 N,
                 embedding_dim,
                 z_dim,
                 hidden_dim,
                 quantization_level):

        super(MlpBlock, self).__init__()

        self.quantization_level = quantization_level

        if quantization_level == 0:
            linears = [nn.Conv1d(z_dim, N, kernel_size=1)]
        else:
            self.embedding = nn.Embedding(num_embeddings=N,
                                          embedding_dim=embedding_dim)
            linears = [nn.Conv1d(z_dim+(embedding_dim*quantization_level), hidden_dim, kernel_size=1)]
            linears = linears + [nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1) for i in range(quantization_level-1)]
            linears = linears + [nn.Conv1d(hidden_dim, N, kernel_size=1)]

        self.linears = nn.ModuleList(linears)

    def forward(self, z, y, embed_ys):
        """
        z: feature vector of shape (n, 256, len_y)
        y: LongTensor (n, len_y) containing predicted (test) or gt (train) indexes of previous level
        embed_ys: (n, (quantization_level-1)*embedding_dim, len_y) embedded ys from previous MlpBlocks,
            or empty tensor, if self.quantization_level <= 2
        return: outputs and embeddings produced so far
        """

        if self.quantization_level > 0:
            y = self.embedding(y)
            y = torch.permute(y, (0, 2, 1))  # shape: (n, embedding_dim, len_y)
            embed_ys = torch.cat((embed_ys, y), dim=1)  # shape: (n, (quantization_level-1)*embedding_dim, len_y)
            z = torch.cat((z, embed_ys), dim=1)  # shape: (n, 256+(reslayer_number*embedding_dim), len_y)

        for layer in self.linears[:-1]:
            z = F.relu(layer(z))
        out = self.linears[-1](z)

        return out, embed_ys


MODEL_CLASS = REVBrain
