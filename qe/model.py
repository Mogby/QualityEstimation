import numpy as np
import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(matrix, dim=None):
  max_score = matrix.max()
  if dim is not None:
    return max_score + \
           torch.log(torch.sum(torch.exp(matrix - max_score), dim=dim))
  else:
    return max_score + \
           torch.log(torch.sum(torch.exp(matrix - max_score)))


class CRF(nn.Module):

  def __init__(self, input_size, num_tags):
    super(CRF, self).__init__()

    self._num_tags = num_tags

    self._input2tags = nn.Linear(input_size, num_tags)

    self._transitions_from_start = nn.Parameter(
      torch.randn(num_tags)
    )
    self._transitions = nn.Parameter(
      torch.randn(num_tags, num_tags)
    )
    self._transitions_to_end = nn.Parameter(
      torch.randn(num_tags)
    )

  def _log_score(self, seq, tags):
    emit_scores = self._input2tags(seq)

    score = self._transitions_from_start[tags[0]] \
            + emit_scores[0, tags[0]]

    for i, item in enumerate(seq[1:], 1):
      score += self._transitions[tags[i], tags[i - 1]]
      score += emit_scores[i, tags[i]]

    score += self._transitions_to_end[tags[-1]]

    return score

  def _partition(self, seq):
    emit_scores = self._input2tags(seq)

    # total log score of all paths ending at given label
    tag_scores_total = self._transitions_from_start \
                       + emit_scores[0]

    for i, item in enumerate(seq[1:], 1):
      # log score of transitions at current step
      transition_scores = self._transitions + tag_scores_total
      tag_scores_total = log_sum_exp(transition_scores, dim=1)
      tag_scores_total += emit_scores[i]

    end_transition_scores = self._transitions_to_end + tag_scores_total

    return log_sum_exp(end_transition_scores)

  def label(self, seq):
    if len(seq.shape) == 3:
      seq = seq[0]

    emit_scores = self._input2tags(seq)

    # maximum log score of a path ending at given label
    tag_scores_total = self._transitions_from_start \
                       + emit_scores[0]

    # previous tag on the best path
    parent = torch.full(
      (len(seq), self._num_tags),
      -1,
      dtype=torch.long
    )

    for i, item in enumerate(seq[1:], 1):
      # log score of transitions at current step
      transition_scores = self._transitions + tag_scores_total
      tag_scores_total, parent[i] = torch.max(transition_scores, dim=1)
      tag_scores_total += emit_scores[i]

    end_transition_scores = self._transitions_to_end + tag_scores_total
    # dim=0 is passed only to retreive the argmax
    path_score, cur_tag = torch.max(end_transition_scores, dim=0)

    path = []
    cur_idx = len(seq) - 1
    while cur_tag != -1:
      # sanity check
      assert cur_idx >= 0

      path.append(cur_tag)
      cur_tag = parent[cur_idx, cur_tag]
      cur_idx -= 1

    # sanity check
    assert len(path) == len(seq)

    path.reverse()
    return torch.tensor(path), path_score

  def log_likelihood(self, seq, tags):
    if len(seq.shape) == 3:
      seq = seq[0]
    if len(tags.shape) == 3:
      tags = tags[0]
    return self._log_score(seq, tags) - self._partition(seq)


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


class EstimatorCRF(nn.Module):

  def __init__(self, embedding_dim, hidden_size, output_size=2, dropout_p=0.2,
               self_attn=True):
    super(EstimatorCRF, self).__init__()

    self._hidden_size = hidden_size
    self._use_self_attn = self_attn

    self._src_enc = EncoderRNN(embedding_dim, hidden_size, dropout_p)
    self._tgt_enc = EncoderRNN(embedding_dim, hidden_size, dropout_p)

    attn_temperature = (1 / (2 * hidden_size)) ** 0.5
    self._attn = ScaledDotProductAttention(attn_temperature)

    features_size = 4 * hidden_size
    if self_attn:
      features_size += 2 * hidden_size

    self._crf = CRF(features_size, output_size)

  def _get_features(self, source, target, training):
    src_features = self._src_enc(source, training)
    tgt_features = self._tgt_enc(target, training)
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

    return torch.cat(estimator_inputs, dim=2)

  def label(self, source, target, training=False):
    features = self._get_features(source, target, training)
    return self._crf.label(features)

  def log_likelihood(self, source, target, labels, training=False):
    features = self._get_features(source, target, training)
    return self._crf.log_likelihood(features, labels)


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
