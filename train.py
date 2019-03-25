#!/usr/bin/env python3

import argparse

import torch
import torch.optim as optim

from torch.utils.data import random_split, DataLoader

from qe.dataset import QEDataset, qe_collate
from qe.embedding import Tokenizer
from qe.model import EstimatorRNN
from qe.train import train


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--src-embeddings', required=True,
                      type=argparse.FileType('rb'))
  parser.add_argument('--mt-embeddings', required=True,
                      type=argparse.FileType('rb'))
  parser.add_argument('--train-path', required=True, type=str)
  parser.add_argument('--dev-path', required=True, type=str)
  parser.add_argument('--batch-size', default=10, type=int)
  parser.add_argument('--num-epochs', default=20, type=int)
  parser.add_argument('--validate-every', default=1, type=int)
  args = parser.parse_args()

  print('Reading src embeddings')
  src_tokenizer = Tokenizer(args.src_embeddings)
  print('Reading mt embeddings')
  mt_tokenizer = Tokenizer(args.mt_embeddings)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(f'Using \'{device}\' device.')

  collate = lambda data: qe_collate(data, device=device)

  train_ds = QEDataset('train', src_tokenizer._word2idx, mt_tokenizer._word2idx,
                      data_dir=args.train_path)
  train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size,
                            collate_fn=collate)

  dev_ds = QEDataset('dev', src_tokenizer._word2idx, mt_tokenizer._word2idx,
                  data_dir=args.dev_path)
  dev_loader = DataLoader(dev_ds, shuffle=True, batch_size=args.batch_size,
                          collate_fn=collate)

  model = EstimatorRNN(150,
                       torch.tensor(src_tokenizer._embeddings),
                       torch.tensor(mt_tokenizer._embeddings),
                       dropout_p=0.2).to(device)

  optimizer = optim.Adadelta(model.parameters(), lr=1)

  train(train_loader, dev_loader, model, optimizer, n_epochs=args.num_epochs,
        validate_every=args.validate_every)


if __name__ == '__main__':
  main()