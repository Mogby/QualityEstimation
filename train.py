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
  parser.add_argument('--learning-rate', default=.1, type=float)
  parser.add_argument('--checkpoint-dir', type=str)
  parser.add_argument('--bert-tokens', action='store_true')
  args = parser.parse_args()

  print('Reading src embeddings')
  src_tokenizer = Tokenizer(args.src_embeddings,
          bert_tokenization=args.bert_tokens)
  print('Reading mt embeddings')
  mt_tokenizer = Tokenizer(args.mt_embeddings,
          bert_tokenization=args.bert_tokens)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(f'Using \'{device}\' device.')

  collate = lambda data: qe_collate(data, device=device)

  train_ds = QEDataset('train', src_tokenizer, mt_tokenizer,
                       use_baseline=True, use_bert_features=True, 
                       data_dir=args.train_path)

  baseline_vocab_sizes = train_ds._baseline_vocab_sizes

  train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size,
                            collate_fn=collate)
  dev_ds = QEDataset('dev', src_tokenizer, mt_tokenizer,
                     use_baseline=True, use_bert_features=True, 
                     data_dir=args.dev_path)
  dev_loader = DataLoader(dev_ds, shuffle=True, batch_size=args.batch_size,
                          collate_fn=collate)

  model = EstimatorRNN(150,
                       torch.tensor(src_tokenizer._embeddings),
                       torch.tensor(mt_tokenizer._embeddings),
                       bert_features_size=768,
                       baseline_vocab_sizes=baseline_vocab_sizes,
                       dropout_p=0.).to(device)

  optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
  train(train_loader, dev_loader, model, optimizer, n_epochs=args.num_epochs,
        validate_every=args.validate_every, save_dir=args.checkpoint_dir)


if __name__ == '__main__':
  main()
