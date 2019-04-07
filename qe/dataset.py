import os
import pickle

import numpy as np
import torch

from torch.utils.data import Dataset

from .embedding import UNK_TOKEN, PAD_TOKEN


OK_TOKEN = 'OK'
BAD_TOKEN = 'BAD'


class QEDataset(Dataset):

  def __init__(self, name, src_tokenizer, mt_tokenizer, use_tags=True,
               use_bert_features=False, use_baseline=False, data_dir=None):
    if data_dir is None:
      data_dir = name

    self._src = self._read_text(
      os.path.join(data_dir, f'{name}.src'),
      src_tokenizer
    )
    self._mt = self._read_text(
      os.path.join(data_dir, f'{name}.mt'),
      mt_tokenizer
    )

    self._use_tags = use_tags
    if use_tags:
      self._word_tags, self._gap_tags = \
          self._read_tags(os.path.join(data_dir, f'{name}.tags'), True)
      src_tag_exts = ['source_tags', 'src_tags']
      for ext in src_tag_exts:
        src_tag_file = os.path.join(data_dir, f'{name}.{ext}')
        if os.path.isfile(src_tag_file):
          break
      self._src_tags = \
          self._read_tags(src_tag_file, False)

    self._use_bert = use_bert_features
    if use_bert_features:
      bert_file = os.path.join(data_dir, f'{name}.bert')
      print(f'Reading {bert_file}')
      with open(bert_file, 'rb') as bert:
        self._bert_features = pickle.load(bert)

    self._use_baseline = use_baseline
    if use_baseline:
      baseline_file = os.path.join(data_dir, f'{name}.baseline')
      print(f'Reading {baseline_file}')
      with open(baseline_file, 'rb') as baseline:
        baseline_dict = pickle.load(baseline)
        self._baseline_features = baseline_dict['features']
        self._baseline_vocab_sizes = baseline_dict['vocab_sizes']

    self._validate()

  def __len__(self):
    return len(self._src)

  def __getitem__(self, idx):
    item = {
      'src': self._src[idx],
      'mt': self._mt[idx],
    }

    if self._use_tags:
      item.update({
        'src_tags': self._src_tags[idx],
        'word_tags': self._word_tags[idx],
        'gap_tags': self._gap_tags[idx],
      })

    if self._use_bert:
      item.update({
        'bert_features': self._bert_features[idx]
      })

    if self._use_baseline:
      item.update({
        'baseline_features': self._baseline_features[idx]
      })

    return item

  def _validate(self):
    num_samples = len(self._src)

    assert len(self._mt) == num_samples
    if self._use_tags:
      assert len(self._src_tags) == num_samples
      assert len(self._word_tags) == num_samples
      assert len(self._gap_tags) == num_samples
    if self._use_bert:
      assert len(self._bert_features) == num_samples
    if self._use_baseline:
      assert len(self._baseline_features) == num_samples
      assert len(self._baseline_vocab_sizes) == \
             len(self._baseline_features[0][0])

    for i in range(num_samples):
      src_len = len(self._src[i])
      mt_len = len(self._mt[i])

      if self._use_tags:
        assert len(self._src_tags[i]) == src_len

        assert len(self._word_tags[i]) == mt_len
        assert len(self._gap_tags[i]) == mt_len + 1

      if self._use_bert:
        assert len(self._bert_features[i]) == mt_len

      if self._use_baseline:
        assert len(self._baseline_features[i]) == mt_len


  def _read_text(self, path, tokenizer):
    print('Reading', path)

    num_unknown = 0

    samples = []
    with open(path, 'r') as file:
      for line in file:
        sample = []

        for word in line.split():
          try:
            token = word2idx[word]
          except:
            token = UNK_TOKEN
            num_unknown += 1
            print('Unknown word:', word)

          sample.append(token)

        samples.append(sample)

    print(num_unknown, 'unknown words encountered')

    return samples

  def _read_tags(self, path, has_gaps):
    print('Reading', path)

    word_tags = []
    if has_gaps:
      gap_tags = []

    with open(path, 'r') as file:
      for line in file:
        line_tags = []
        for tag in line.split():
          if tag == OK_TOKEN:
            line_tags.append(1)
          elif tag == BAD_TOKEN:
            line_tags.append(0)
          else:
            raise Exception

        if has_gaps:
          word_tags.append(line_tags[1::2])
          gap_tags.append(line_tags[::2])
        else:
          word_tags.append(line_tags)

    if has_gaps:
      return word_tags, gap_tags

    return word_tags


def qe_collate(data, device=torch.device('cpu')):
  def _pad_sequence(sequence):
    sequence = [item.copy() for item in sequence]

    lens = [len(item) for item in sequence]
    max_len = max(lens)
    min_len = min(lens)

    if max_len == min_len:
        return sequence

    pad_item = np.full_like(sequence[0][0], PAD_TOKEN)

    for i, item in enumerate(sequence):
      pad_amount = max_len - len(item)
      if pad_amount > 0:
          pad_array = np.asarray([pad_item] * pad_amount)
          sequence[i] = np.concatenate((item, pad_array))

    return sequence

  merged = {}
  for key in data[0].keys():
    sequence = [sample[key] for sample in data]
    merged[key] = torch.tensor(_pad_sequence(sequence)).transpose(0, 1).to(device)

  return merged


# BERT section


def tokenize_sentence(sent, tokenizer):
  tokens = ['[CLS]']
  indices = [-1]
  for i, word in enumerate(sent.split()):
    bert_tokens = tokenizer.tokenize(word)
    tokens += bert_tokens
    indices += [i] * len(bert_tokens)
  tokens.append('[SEP]')
  indices.append(-2)
  return tokenizer.convert_tokens_to_ids(tokens), indices


def merge_sentences(toks1, idx1,
                    toks2, idx2):
  toks = toks1 + toks2[1:]
  idx = idx1 + idx2[1:]
  idx[-1] = -3
  return toks, idx


def detokenize(tokens, tokenizer):
  return [tokenizer.ids_to_tokens[token] for token in tokens]


class BertQEDataset(Dataset):
  def __init__(self, name, data_dir, tokenizer):
    self._tokenizer = tokenizer

    print('Reading src')
    src, src_indices = self._read_samples(
      os.path.join(data_dir, f'{name}.src')
    )
    print('Reading mt')
    mt, mt_indices = self._read_samples(
      os.path.join(data_dir, f'{name}.mt')
    )
    print('Reading tags')
    tags = self._read_tags(os.path.join(data_dir, f'{name}.tags'))

    print('Merging data')
    self._src, self._indices, self._segs, self._mt_mask, self._tags = self._merge_data(
      src, src_indices,
      mt, mt_indices,
      tags
    )

    print('Validating')
    assert (self._is_valid())

  def __len__(self):
    return len(self._src)

  def __getitem__(self, idx):
    return {
      'src': self._src[idx],
      'indices': self._indices[idx],
      'segs': self._segs[idx],
      'mt_mask': self._mt_mask[idx],
      'tags': self._tags[idx],
    }

  def _is_valid(self):
    if len(self._src) != len(self._indices) \
            or len(self._src) != len(self._tags) \
            or len(self._src) != len(self._segs):
      return False
    return True

  def _merge_data(self, srcs, src_idxs, mts, mt_idxs, tags):
    src_merged = []
    idx_merged = []
    seg_merged = []
    tag_merged = []
    mt_mask = []

    max_src_len = 0

    # merging samples
    for src, src_idx, mt, mt_idx, tag in zip(srcs, src_idxs, mts, mt_idxs, tags):
      new_src, new_idx = merge_sentences(
        src, src_idx,
        mt, mt_idx
      )
      new_tag = [0] * len(src) + [tag[2 * i + 1] for i in mt_idx[1:-1]] + [0]

      src_merged.append(np.array(new_src, dtype=np.int64))
      idx_merged.append(np.array(new_idx, dtype=np.int64))
      tag_merged.append(np.array(new_tag, dtype=np.int64))

      mt_begin = new_idx.index(-2) + 1
      seg_merged.append(np.array(
        [0] * mt_begin + [1] * (len(new_src) - mt_begin),
        dtype=np.int64
      ))

      mt_mask.append(np.array(seg_merged[-1][:], dtype=np.int64))
      mt_mask[-1][-1] = 0

    return src_merged, idx_merged, seg_merged, mt_mask, tag_merged

  def _read_samples(self, path):
    samples = []
    indices = []
    with open(path, 'r') as file:
      for line in file:
        toks, idx = tokenize_sentence(line, self._tokenizer)
        samples.append(toks)
        indices.append(idx)

    return samples, indices

  def _read_tags(self, path):
    tags = []
    with open(path, 'r') as file:
      for line in file:
        line_tags = []
        for tag in line.split():
          if tag == OK_TOKEN:
            line_tags.append(1)
          elif tag == BAD_TOKEN:
            line_tags.append(0)
          else:
            raise Exception
        tags.append(line_tags)

    return tags
