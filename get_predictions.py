#!/usr/bin/env python3

import argparse

import torch

from torch.utils.data import random_split, DataLoader

from qe.dataset import QEDataset, qe_collate
from qe.embedding import Tokenizer
from qe.model import EstimatorRNN
from qe.train import validate


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset-path', required=True, type=str)
  parser.add_argument('--dataset-name', required=True, type=str)
  parser.add_argument('--src-embeddings', required=True,
                      type=argparse.FileType('rb'))
  parser.add_argument('--mt-embeddings', required=True,
                      type=argparse.FileType('rb'))
  parser.add_argument('--model-file', required=True, type=str)
  args = parser.parse_args()

  print('Reading src embeddings')
  src_tokenizer = Tokenizer(args.src_embeddings)
  print('Reading mt embeddings')
  mt_tokenizer = Tokenizer(args.mt_embeddings)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(f'Using \'{device}\' device.')

  data = QEDataset(args.dataset_name, src_tokenizer._word2idx,
                   mt_tokenizer._word2idx,
                   use_tags=True,
                   use_baseline=True,
                   data_dir=args.dataset_path)
  collate = lambda data: qe_collate(data, device=device)
  data_loader = DataLoader(data, collate_fn=collate)

  baseline_vocab_sizes = data._baseline_vocab_sizes
  model = EstimatorRNN(150,
                       torch.tensor(src_tokenizer._embeddings),
                       torch.tensor(mt_tokenizer._embeddings),
                       baseline_vocab_sizes=baseline_vocab_sizes,
                       dropout_p=0.).to(device)
  model.load_state_dict(torch.load(args.model_file, map_location='cpu'))

  print(validate(data_loader, model))
  return

  METHOD_NAME = 'TEST'
  for i, item in enumerate(data_loader):
    _, mt_tag = model.predict(item['src'], item['mt'])
    mt = mt_tokenizer.tokens_to_sentence(item['mt'][:, 0]).split()
    for j, (word, tag) in enumerate(zip(mt, mt_tag[1::2, 0])):
      tag = 'OK' if tag == 1 else 'BAD'

      # <METHOD NAME> <TYPE> <SEGMENT NUMBER> <WORD INDEX> <WORD> <BINARY SCORE>
      print(METHOD_NAME, 'mt', i, j, word, tag)


if __name__ == '__main__':
  main()