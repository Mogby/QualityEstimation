import numpy as np
import onmt
import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertModel

from .embedding import PAD_TOKEN


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


class BaselineFeatureConverter(nn.Module):
  '''
  Converts categorial baseline features to one-hot vectors.

  Receives a K baseline features and converts them to float vector of size L.
  Operates on minibatches of size N.
  '''

  def __init__(self, vocab_sizes):
    super(BaselineFeatureConverter, self).__init__()

    self._num_features = len(vocab_sizes)
    self._embeds = [None] * self._num_features
    self._vocab_sizes = vocab_sizes[:]
    self._features_size = 0
    for i, size in enumerate(vocab_sizes):
      if size == -1:
        self._features_size += 1
        continue

      self._features_size += size
      one_hot_embeds = torch.eye(size)
      unk_embed = torch.zeros(size)
      self._embeds[i] = nn.Embedding.from_pretrained(
        torch.cat([one_hot_embeds, unk_embed.unsqueeze(0)])
      ).to(device)

  def forward(self, features):
    '''
    Converts baseline features to one-hot.

    :param features: Batch of features of shape (N, K)
    :return: Batch of converted features of shape (N, L)
    '''
    N, K = features.shape
    features = features.view(-1, K)
    converted = []
    for i in range(self._num_features):
      column = features[:, i]
      if self._embeds[i] is not None:
        column[column < 0] = self._vocab_sizes[i]
        column = self._embeds[i](column.to(torch.long))
      else:
        column = column.unsqueeze(1)
      converted.append(column)
    converted = torch.cat(converted, dim=1)
    return converted.reshape(N, -1)


class ONMTEstimator(nn.Module):

  def __init__(self, hidden_size, src_embeddings, mt_embeddings):
    super(ONMTEstimator, self).__init__()

    self._hidden_size = hidden_size
    self._emb_dim = src_embeddings.shape[1]
    assert mt_embeddings.shape[1] == self._emb_dim

    src_emb = onmt.modules.Embeddings(
        self._emb_dim,
        len(src_embeddings),
        PAD_TOKEN,
    )
    # src_emb.load_pretrained_vectors('./data/precalc/embeddings/en.pth')
    # src_emb.word_lut.weight.data = src_embeddings.to(device)
    # src_emb.word_lut.weight.requires_grad = False

    self._src_enc = onmt.encoders.RNNEncoder(
        hidden_size=hidden_size,
        num_layers=1,
        rnn_type='LSTM',
        bidirectional=True,
        embeddings=src_emb,
        dropout=0.
    )

    mt_emb = onmt.modules.Embeddings(
        self._emb_dim,
        len(mt_embeddings),
        PAD_TOKEN,
    )
    # mt_emb.load_pretrained_vectors('./data/precalc/embeddings/de.pth')
    # mt_emb.word_lut.weight.data = mt_embeddings.to(device)
    # mt_emb.word_lut.weight.requires_grad = False

    self._mt_enc = onmt.encoders.RNNEncoder(
        hidden_size=hidden_size,
        num_layers=1,
        rnn_type='LSTM',
        bidirectional=True,
        embeddings=mt_emb,
        dropout=0.
    )

    self._attn = onmt.modules.MultiHeadedAttention(
        head_count=1,
        model_dim=hidden_size,
        dropout=0.,
    )

    self._out = nn.Linear(2*hidden_size, 2)
    self._loss = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

  def forward(self, src, mt, **kwargs):
    max_src_len, batch_len = src.shape
    max_mt_len = mt.shape[0]

    features = []

    nsrc = src.clone().unsqueeze(2)
    nsrc[nsrc < 0] = 0
    src_feats = self._src_enc(nsrc)[1].transpose(0, 1)
    nmt = mt.clone().unsqueeze(2)
    nmt[nmt < 0] = 0
    mt_feats = self._mt_enc(nmt)[1].transpose(0, 1)
    features.append(mt_feats)

    attn_mask = torch.empty(batch_len, max_mt_len, max_src_len,
                            dtype=torch.uint8, device=device)
    for i in range(batch_len):
      attn_mask[i,:,:] = (src[:,i] != PAD_TOKEN)
    context, _ = self._attn(src_feats, src_feats, mt_feats, attn_mask)
    features.append(context)

    scores = self._out(
        torch.cat(features, dim=2)
    )

    return scores.transpose(0, 1)

  def loss(self, src, mt, aligns, src_tags=None,
           word_tags=None, gap_tags=None, **kwargs):
    scores = self(src, mt, **kwargs)
    # print(scores.shape)
    # print(word_tags.shape)
    return self._loss(scores.transpose(1, 2), word_tags)

  def predict(self, src, mt, aligns, **kwargs):
    with torch.no_grad():
      src_tags = torch.ones_like(src)
      word_tags = torch.ones_like(mt)
      gap_tags = torch.ones((mt.shape[0] + 1,) + mt.shape[1:])

      scores = self(src, mt)

      batch_len = src.shape[1]
      for i in range(batch_len):
        word_tags[:, i] = (scores[:,i,1] > scores[:,i,0])
        for j_src, j_mt in aligns[:, i]:
          if j_mt == -1:
            break
          if word_tags[j_mt, i] == 0:
            src_tags[j_src, i] = 0

      word_tags[mt == PAD_TOKEN] = -1

      return src_tags, word_tags, gap_tags


class EstimatorRNN(nn.Module):

  def __init__(self, hidden_size, src_embeddings, mt_embeddings,
               output_size=2, bert_features_size=0, baseline_vocab_sizes=None,
               predict_gaps=True, dropout_p=0.2,
               use_confidence=True, self_attn=True):
    super(EstimatorRNN, self).__init__()

    self._hidden_size = hidden_size
    self._use_self_attn = self_attn

    self._emb_dim = src_embeddings.shape[1]
    assert mt_embeddings.shape[1] == self._emb_dim

    # self._src_emb = nn.Embedding.from_pretrained(src_embeddings)
    # self._tgt_emb = nn.Embedding.from_pretrained(mt_embeddings)
    self._src_emb = nn.Embedding(*src_embeddings.shape)
    self._tgt_emb = nn.Embedding(*mt_embeddings.shape)

    features_size = 0

    if hidden_size > 0:
      self._src_enc = EncoderRNN(self._emb_dim, hidden_size, dropout_p)
      self._tgt_enc = EncoderRNN(self._emb_dim, hidden_size, dropout_p)

      attn_temperature = (1 / (2 * hidden_size)) ** 0.5
      self._attn = ScaledDotProductAttention(attn_temperature)

      features_size += 4 * hidden_size

      if self_attn:
        features_size += 2 * hidden_size
    else:
      assert not self_attn
      assert not use_confidence

    self._use_confidence = use_confidence
    if use_confidence:
      features_size += 3

    features_size +=  bert_features_size

    if baseline_vocab_sizes is not None:
      self._baseline_converter = BaselineFeatureConverter(baseline_vocab_sizes)
      features_size += self._baseline_converter._features_size

    self._crf = CRF(features_size, output_size)
    self._out = nn.Linear(features_size, output_size)
    self._loss = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    self._predict_gaps = predict_gaps
    if predict_gaps:
      self._gap_crf = CRF(2 * features_size, output_size)

    self._zero_emb = torch.zeros(self._emb_dim)
    self._zero_features = torch.zeros(features_size)

  def _convert_baseline_features(self, baseline_features):
    N, M, K = baseline_features.shape
    baseline_features = baseline_features.reshape(-1, K)
    converted = self._baseline_converter(baseline_features).to(device)
    return converted.reshape(N, M, -1)

  def _make_gap_features(self, mt_features, mt_inds):
    nxt_features = [mt_features[0].unsqueeze(0)]
    prv_features = [self._zero_features.unsqueeze(0).to(device)]
    for i, (prv, nxt) in enumerate(zip(mt_inds[:-1], mt_inds[1:])):
      if prv != nxt:
       nxt_features.append(mt_features[i+1].unsqueeze(0))
       prv_features.append(mt_features[i].unsqueeze(0))
    nxt_features.append(self._zero_features.unsqueeze(0).to(device))
    prv_features.append(mt_features[-1].unsqueeze(0))

    nxt_features = torch.cat(nxt_features)
    prv_features = torch.cat(prv_features)

    return torch.cat([nxt_features, prv_features], dim=1)


  def forward(self, src, mt, bert_features=None, baseline_features=None,
              training=True):
    '''
    src: shape (SRC, N)
    mt: shape (MT, N)
    bert_features: shape (MT, N, BERT)
    baseline_features: shape (MT, N, BASELINE)
    '''
    max_src_len, batch_len = src.shape
    max_tgt_len = mt.shape[0]

    features = []

    if self._hidden_size > 0:
      src_emb = torch.empty(max_src_len, batch_len, self._emb_dim).to(device) # shape (SRC, N, EMB)
      src_emb[src >= 0] = self._src_emb(src[src >= 0])
      if (src < 0).any():
        src_emb[src < 0] = self._zero_emb.to(device)
      src_features = self._src_enc(src_emb, training=training).transpose(0, 1) # shape (SRC, N, HIDDEN)

      tgt_emb = torch.empty(max_tgt_len, batch_len, self._emb_dim).to(device) # shape (N, MT, EMB)
      tgt_emb[mt >= 0] = self._tgt_emb(mt[mt >= 0])
      if (mt < 0).any():
        tgt_emb[mt < 0] = self._zero_emb.to(device)
      tgt_features = self._tgt_enc(tgt_emb, training=training).transpose(0, 1) # shape (N, MT, HIDDEN)


      features.append(tgt_features)

      attn_mask = torch.zeros(batch_len, len(mt), len(src),
                              dtype=torch.uint8, device=device) # shape (N, MT, SRC)
      for i in range(batch_len):
        attn_mask[i,:,:,] = (src[:,i] != PAD_TOKEN)

      context, align = self._attn(tgt_features,
                                  src_features,
                                  src_features,
                                  attn_mask) # shape (N, MT, EMB), (N, MT, SRC)
      features.append(context)

    if self._use_confidence:
      topk, _ = torch.topk(align, 2, dim=2) # shape (N, MT, 2)
      features.append(topk)

      non_zero_mask = topk[:,:,0] > 0
      confidence = topk[:,:,1] # shape (N, MT)
      confidence[non_zero_mask] /= topk[:,:,0][non_zero_mask]
      features.append(confidence.unsqueeze(2))

    self_align = None
    if self._use_self_attn:
      self_attn_mask = torch.diag(torch.tensor(
        [True] * max_tgt_len,
        device=device
      )) # shape (MT, MT)
      self_context, self_align = self._attn(tgt_features,
                                            tgt_features,
                                            tgt_features,
                                            self_attn_mask) # shape (N, MT, EMB), (N, MT, MT)
      features.append(self_context)

    if bert_features is not None:
      features.append(bert_features.transpose(0, 1))

    if baseline_features is not None:
      baseline_features = self._convert_baseline_features(baseline_features)
      features.append(baseline_features.transpose(0, 1))

    features = torch.cat(features, dim=2) # shape (N, MT, FEATS)
    return features.transpose(0, 1), None # shape (MT, N, FEATS), (,)

  def loss(self, src, src_inds, mt, mt_inds, aligns, src_tags=None, word_tags=None, 
           gap_tags=None, **kwargs):
    features, align = self.forward(src, mt, **kwargs)

    batch_len = src.shape[1]
    loss = 0
    for i in range(batch_len):
      tokens_len = (mt_inds[:,i] >= 0).sum()
      mt_len = mt_inds[:,i].max() + 1

      loss -= self._crf.log_likelihood(features[:tokens_len,i,:],
                                       word_tags[:tokens_len,i])

      if self._predict_gaps:
        gap_features = self._make_gap_features(
            features[:tokens_len,i,:], mt_inds[:tokens_len, i]
        )
        assert len(gap_features) == mt_len+1
        loss -= self._gap_crf.log_likelihood(gap_features,
                                             gap_tags[:mt_len+1,i])

    return loss / batch_len

  def predict(self, src, src_inds, mt, mt_inds, aligns, **kwargs):
    kwargs.setdefault('training', False)

    with torch.no_grad():
      src_tags = torch.ones_like(src)
      word_tags = torch.ones_like(mt)
      gap_tags = torch.ones((mt_inds.max() + 2,) + mt.shape[1:])

      features, align = self.forward(src, mt, **kwargs)

      batch_len = src.shape[1]
      for i in range(batch_len):
        tokens_len = (mt_inds[:,i] >= 0).sum()
        mt_len = mt_inds[:,i].max() + 1

        expanded_tags, _ = self._crf.label(features[:tokens_len,i,:])
        for j in range(mt_len):
          word_tags[j, i] = (expanded_tags[mt_inds[:tokens_len, i] == j] == 1).all()
        for i_src, i_mt in aligns[:, i]:
          if i_mt == -1:
            break
          if word_tags[i_mt, i] == 0:
            src_tags[i_src, i] = 0

        if self._predict_gaps:
          gap_features = self._make_gap_features(
              features[:tokens_len,i,:], mt_inds[:tokens_len, i]
          )
          assert len(gap_features) == mt_len+1
          gap_tags[:mt_len+1, i], _ = self._gap_crf.label(gap_features)

      return src_tags, word_tags, gap_tags


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
