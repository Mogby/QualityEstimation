import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Borrowed from
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/
class ScaledDotProductAttention(nn.Module):

  def __init__(self, temperature, attn_dropout=0.1):
    super().__init__()
    self._temperature = temperature
    self._dropout = nn.Dropout(attn_dropout)
    self._softmax = nn.Softmax(dim=2)

  def forward(self, q, k, v, mask=None):
    align = torch.bmm(q, k.transpose(1, 2))
    align = align / self._temperature

    if mask is not None:
      align = align.masked_fill(mask, -np.inf)

    align = self._softmax(align)
    attn = torch.bmm(align, v)

    return attn, align


class EncoderRNN(nn.Module):

  def __init__(self, embedding_dim, hidden_size, dropout_p=0.2):
    super(EncoderRNN, self).__init__()
    self._hidden_size = hidden_size
    self._dropout = nn.Dropout(p=dropout_p)
    self._gru = nn.GRU(embedding_dim, hidden_size,
                       batch_first=True, bidirectional=True)

  def forward(self, input, training=False):
    if len(input.shape) == 2:
      input = input.unsqueeze(0)

    if training:
      input = self._dropout(input)
    hidden = torch.zeros(2, 1, self._hidden_size).to(device)
    output, hidden = self._gru(input, hidden)
    return output


class EstimatorRNN(nn.Module):

  def __init__(self, embedding_dim, hidden_size, output_size=2,
               dropout_p=0.2, self_attn=True):
    super(EstimatorRNN, self).__init__()

    self._hidden_size = hidden_size
    self._use_self_attn = self_attn

    self._src_enc = EncoderRNN(embedding_dim, hidden_size, dropout_p)
    self._tgt_enc = EncoderRNN(embedding_dim, hidden_size, dropout_p)

    attn_temperature = (1 / (2 * hidden_size)) ** 0.5
    self._attn = ScaledDotProductAttention(attn_temperature)

    features_size = 4 * hidden_size
    if self_attn:
      features_size += 2 * hidden_size

    self._out = nn.Linear(features_size, output_size)
    self._softmax = nn.LogSoftmax(dim=2)

  def forward(self, source, target, training=False):
    if len(source.shape) == 2:
      source = source.unsqueeze(0)
    if len(target.shape) == 2:
      target = target.unsqueeze(0)

    src_features = self._src_enc(source)
    tgt_features = self._tgt_enc(target)
    estimator_inputs = [
      tgt_features,
    ]

    context, align = self._attn(tgt_features, src_features, src_features)
    estimator_inputs.append(context)

    self_align = None
    if self._use_self_attn:
      self_align_mask = torch.diag(torch.tensor(
        [True] * target.shape[1],
        device=device
      ))
      self_context, self_align = self._attn(tgt_features,
                                            tgt_features,
                                            tgt_features,
                                            self_align_mask)
      estimator_inputs.append(self_context)

    output = self._softmax(
      self._out(
        torch.cat(
          estimator_inputs,
          dim=2
        )
      )
    )

    return output, align, self_align