import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from probsparse_mhsa import ProbAttention, FullAttention, AttentionLayer


class Informer(nn.Module):
    def __init__(self, d_model, num_head, len_seq, factor=5):
        super(Informer, self).__init__()

        self.positional_encoding_layer = PositionalEncoding(embedding_dim=d_model, len_seq=len_seq)

        # Attention
        Attn = ProbAttention(mask_flag=False, factor=factor)
        self.MHSA_layer = AttentionLayer(Attn, d_model, num_head=num_head)

        # encoder
        self.encoder = EncoderLayer(attention=self.MHSA_layer, d_model=d_model, d_ff=2048, dropout=0.1,
                                    activation="relu")

        # decoder
        self.decoder = DecoderLayer(prob_attention=self.MHSA_layer, d_model=d_model, d_ff=2048, dropout=0.1,
                                    activation="relu")

        self.distilling_layer = DistillLayer(d_model=d_model)

        self.out_projection = nn.Linear(in_features=d_model, out_features=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_data):
        enc_inp = self.positional_encoding_layer(input_data)
        dec_inp = self.positional_encoding_layer(input_data)

        # encoder
        enc_layer1 = self.distilling_layer(self.encoder(enc_inp))
        enc_layer2 = self.distilling_layer(self.encoder(enc_layer1))
        enc_layer3 = self.distilling_layer(self.encoder(enc_layer2))
        # enc_layer4 = self.encoder(enc_layer3)

        # decoder
        dec_layer1 = self.decoder(dec_inp, enc_layer3)

        out = self.out_projection(dec_layer1)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, len_seq):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros((len_seq, embedding_dim)).float()
        pe.require_grad = False
        position = torch.arange(0, len_seq).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = x + self.dropout(self.attention(x, x, x))
        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class DistillLayer(nn.Module):
    def __init__(self, d_model):
        super(DistillLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=d_model,
                                  out_channels=d_model,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, prob_attention, d_model, d_ff, dropout, activation):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.prob_attention = prob_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross):
        out1 = x + self.dropout(self.prob_attention(x, x, x))

        out2 = out1 + self.dropout(self.prob_attention(cross, out1, out1))

        out3 = self.dropout(self.activation(self.conv1(out2.transpose(-1, 1))))
        out3 = self.dropout(self.conv2(out3).transpose(-1, 1))
        out3 = out3 + out2

        return out3
