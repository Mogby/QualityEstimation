import os

import numpy as np
import torch

from torch.utils.data import Dataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


GAP_TOKEN = '_'
OK_TOKEN = 'OK'
BAD_TOKEN = 'BAD'
UNK_TOKEN = '-'


class QEDataset(Dataset):

  def __init__(self, name, w2v_src, w2v_mt, spoil_p=0.1, data_dir=None):
    if data_dir is None:
      data_dir = name

    self._src = self._read_corpus(
      os.path.join(data_dir, f'{name}.src'),
      w2v_src,
    )
    self._mt = self._read_corpus(
      os.path.join(data_dir, f'{name}.mt'),
      w2v_mt,
      add_gaps=True,
    )
    self._tags = self._read_tags(os.path.join(data_dir, f'{name}.tags'))

    self._vec_dim = self._src[0].shape[1]
    self._spoil_p = spoil_p

    assert (self._is_valid())

  def __len__(self):
    return len(self._src)

  def __getitem__(self, idx):
    #mt, tags = self._spoil_mt(self._mt[idx], self._tags[idx])
    mt, tags = self._mt[idx], self._tags[idx]
    return {
      'src': self._src[idx],
      'mt': mt,
      'tags': tags,
    }

  def _is_valid(self):
    if len(self._src) != len(self._mt):
      return False
    if len(self._src) != len(self._tags):
      return False

    for mt, tags in zip(self._mt, self._tags):
      if len(mt) != len(tags):
        return False

    return True

  def _get_random_mt_word_vecs(self, n_vecs):
    return torch.randn(n_vecs, self._vec_dim, dtype=torch.float, device=device)

  def _spoil_mt(self, mt, tags):
    mt_spoiled = mt.clone()
    tags_spoiled = tags.clone()
    spoil_mask = torch.rand(len(mt), device=device) < self._spoil_p
    spoil_mask *= tags_spoiled.to(torch.uint8)  # spoil only ok tokens

    mt_spoiled[spoil_mask] = self._get_random_mt_word_vecs(spoil_mask.sum())
    tags_spoiled[spoil_mask] = 0

    return mt_spoiled, tags_spoiled

  def _read_corpus(self, path, w2v, add_gaps=False):
    gap_vec = w2v[GAP_TOKEN]
    num_unknown = 0

    samples = []
    with open(path, 'r') as file:
      for line in file:
        sample = [gap_vec] if add_gaps else []

        for word in line.split():
          try:
            word_vec = w2v[word]
          except:
            word_vec = w2v[UNK_TOKEN]
            num_unknown += 1
            print('Unknown word:', word)

          sample.append(word_vec)
          if add_gaps:
            sample.append(gap_vec)

        sample = torch.tensor(sample, dtype=torch.float32,
                              device=device)
        samples.append(sample)

    print('num_unknown:', num_unknown)

    return samples

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
        line_tags = torch.tensor(line_tags, device=device)
        tags.append(line_tags)

    return tags


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

    self._src = np.array(self._src, dtype=np.int64)
    self._indices = np.array(self._indices, dtype=np.int64)
    self._segs = np.array(self._segs, dtype=np.int64)
    self._mt_mask = np.array(self._mt_mask, dtype=np.float64)
    self._tags = np.array(self._tags, dtype=np.int64)

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

      src_merged.append(new_src)
      idx_merged.append(new_idx)
      tag_merged.append(new_tag)

      mt_begin = new_idx.index(-2) + 1
      seg_merged.append([0] * mt_begin
                        + [1] * (len(new_src) - mt_begin))

      mt_mask.append(seg_merged[-1][:])
      mt_mask[-1][-1] = 0

      max_src_len = max(max_src_len, len(new_src))

    # padding samples
    for src, idx, seg, mask, tag in zip(src_merged, idx_merged, seg_merged, mt_mask, tag_merged):
      pad_len = max_src_len - len(src)
      src.extend([0] * pad_len)
      idx.extend([0] * pad_len)
      seg.extend([0] * pad_len)
      mask.extend([0] * pad_len)
      tag.extend([0] * pad_len)

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