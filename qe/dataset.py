import os

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
