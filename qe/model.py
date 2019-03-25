import numpy as np
import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Borrowed from
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/
class ScaledDotProductAttention(nn.Module):
  '''
  Computes scaled dot product attention and attention weights. Supports masking.

  Receives a Q query vectors of size D and K keys and values of size D.
  Operates on minibatches of size N.
  '''

  def __init__(self, temperature, attn_dropout=0.1):
    super().__init__()
    self._temperature = temperature
    self._dropout = nn.Dropout(attn_dropout)
    self._softmax = nn.Softmax(dim=2)

  def forward(self, queries, keys, values, mask=None):
    '''
    Compute scaled dot product attention and attention weights.

    :param queries: Batch of queries of shape (N, Q, D)
    :param keys: Batch of keys of shape (N, K, D)
    :param values: Batch of values of shape (N, K, D)
    :param mask: Batch of binary masks of shape (N, Q, K). If `mask[i, j, k]`
    is `1`, then query `queries[i, j]` will not pay attention to key `keys[i, k]`.
    :return: tuple of attention batch of shape (N, Q, D) and batch of attention
    weights of shape (N, Q, K)
    '''
    weights = torch.bmm(queries, keys.transpose(1, 2))
    weights = weights / self._temperature

    if mask is not None:
      weights = weights.masked_fill(mask, -np.inf)

    weights = self._softmax(weights)
    attention = torch.bmm(weights, values)

    return attention, weights


class EncoderRNN(nn.Module):

  def __init__(self, embedding_dim, hidden_size, dropout_p=0.2):
    super(EncoderRNN, self).__init__()
    self._hidden_size = hidden_size
    self._dropout = nn.Dropout(p=dropout_p)
    self._gru = nn.GRU(embedding_dim, hidden_size, bidirectional=True)

  def forward(self, input, training=True):
    if training:
      input = self._dropout(input)

    output, hidden = self._gru(input)
    return output


class EstimatorRNN(nn.Module):

  def __init__(self, hidden_size, src_embeddings, mt_embeddings,
               output_size=2, dropout_p=0.2, self_attn=True):
    super(EstimatorRNN, self).__init__()

    self._hidden_size = hidden_size
    self._use_self_attn = self_attn

    self._emb_dim = src_embeddings.shape[1]
    assert mt_embeddings.shape[1] == self._emb_dim

    self._src_emb = nn.Embedding.from_pretrained(src_embeddings)
    self._tgt_emb = nn.Embedding.from_pretrained(mt_embeddings)

    self._src_enc = EncoderRNN(self._emb_dim, hidden_size, dropout_p)
    self._tgt_enc = EncoderRNN(self._emb_dim, hidden_size, dropout_p)

    attn_temperature = (1 / (2 * hidden_size)) ** 0.5
    self._attn = ScaledDotProductAttention(attn_temperature)

    features_size = 4 * hidden_size
    if self_attn:
      features_size += 2 * hidden_size

    self._out = nn.Linear(features_size, output_size)
    self._softmax = nn.LogSoftmax(dim=2)

    self._loss = nn.NLLLoss(reduction='none')

    self._zero_emb = torch.zeros(self._emb_dim)

  def forward(self, src, mt, training=True):
    max_src_len, batch_len = src.shape
    max_tgt_len = mt.shape[0]

    device = src.device

    src_emb = torch.empty(max_src_len, batch_len, self._emb_dim).to(device)
    src_emb[src >= 0] = self._src_emb(src[src >= 0])
    if (src < 0).any():
      src_emb[src < 0] = self._zero_emb.to(device)
    src_features = self._src_enc(src_emb, training=training).transpose(0, 1)

    tgt_emb = torch.empty(max_tgt_len, batch_len, self._emb_dim).to(device)
    tgt_emb[mt >= 0] = self._tgt_emb(mt[mt >= 0])
    if (mt < 0).any():
      tgt_emb[mt < 0] = self._zero_emb.to(device)
    tgt_features = self._tgt_enc(tgt_emb, training=training).transpose(0, 1)

    estimator_inputs = [
      tgt_features,
    ]

    attn_mask = torch.zeros(batch_len, len(mt), len(src),
                            dtype=torch.uint8, device=device)
    if (src < 0).any():
      attn_mask[:] = (src < 0).transpose(0, 1).unsqueeze(1)
    context, align = self._attn(tgt_features,
                                src_features,
                                src_features,
                                attn_mask)
    estimator_inputs.append(context)

    self_align = None
    if self._use_self_attn:
      self_attn_mask = torch.diag(torch.tensor(
        [True] * mt.shape[0],
        device=device
      ))
      self_context, self_align = self._attn(tgt_features,
                                            tgt_features,
                                            tgt_features,
                                            self_attn_mask)
      estimator_inputs.append(self_context)

    estimator_inputs = torch.cat(estimator_inputs, dim=2)

    scores = self._softmax(self._out(estimator_inputs)).transpose(0, 1)

    return scores, align

  def loss(self, src, mt, src_tags=None, word_tags=None, gap_tags=None,
           training=True):
    scores, align = self.forward(src, mt, training=training)

    losses = self._loss(scores.transpose(1, 2), word_tags)
    loss_mask = (mt >= 0).to(torch.float)
    loss = (losses * loss_mask).sum(dim=0).mean()

    return loss

  def predict(self, src, mt):
    src_tags = torch.ones_like(src)
    mt_tags = torch.ones((2 * len(mt) + 1,) + mt.shape[1:])

    scores, align = self.forward(src, mt, training=False)

    mt_tags[1::2][scores[:,:,0] >= np.log(0.5)] = 0

    return src_tags, mt_tags


class BertQE(nn.Module):
  def __init__(self, bert_hidden_size=768):
    super(BertQE, self).__init__()

    self._bert = BertModel.from_pretrained('bert-base-multilingual-cased')
    self._out = nn.Linear(bert_hidden_size, 2)
    self._softmax = nn.LogSoftmax(dim=2)

  def forward(self, input_ids, seg_ids):
    bert_out, _ = self._bert(input_ids, seg_ids,
                             output_all_encoded_layers=False)
    out = self._out(bert_out)
    return self._softmax(out)
