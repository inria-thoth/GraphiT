# -*- coding: utf-8 -*-
import torch
from torch import nn
from .layers import DiffTransformerEncoderLayer


class GraphTransformer(nn.Module):
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 lap_pos_enc=False, lap_pos_enc_dim=0):
        super(GraphTransformer, self).__init__()

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            # We embed the pos. encoding in a higher dim space
            # as Bresson et al. and add it to the features.
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nb_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalAvg1D()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
        )

    def forward(self, x, masks, x_pe, x_lap_pos_enc=None, degree=None):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x = x.permute(1, 0, 2)
        output = self.embedding(x)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output = self.encoder(output, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)
        # we make sure to correctly take the masks into account when pooling
        output = self.pooling(output, masks)
        # we only do mean pooling for now.
        return self.classifier(output)


class DiffTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, pe, degree=None, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, pe=pe, degree=degree, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DiffGraphTransformer(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0):
        super(DiffGraphTransformer, self).__init__()

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)
        encoder_layer = DiffTransformerEncoderLayer(
                d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        self.encoder = DiffTransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalAvg1D()
        #self.classifier = nn.Linear(in_features=d_model,
        #                            out_features=nb_class, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
            )

    def forward(self, x, masks, pe, x_lap_pos_enc=None, degree=None):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x = x.permute(1, 0, 2)
        output = self.embedding(x)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output = self.encoder(output, pe, degree=degree, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)
        # we make sure to correctly take the masks into account when pooling
        output = self.pooling(output, masks)
        # we only do mean pooling for now.
        return self.classifier(output)


class GlobalAvg1D(nn.Module):
    def __init__(self):
        super(GlobalAvg1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask = (~mask).float().unsqueeze(-1)
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1)
