#!/usr/bin/env python3

import argparse

import torch
import torch.optim as optim

from torch.utils.data import random_split, DataLoader

from qe.dataset import QEDataset, qe_collate
from qe.embedding import Tokenizer
from qe.model import device, QualityEstimator
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
  parser.add_argument('--predict-gaps', action='store_true')
  parser.add_argument('--use-bert', action='store_true')
  parser.add_argument('--use-baseline', action='store_true')
  parser.add_argument('--self-attention', action='store_true')
  parser.add_argument('--hidden-size', default=300, type=int)
  parser.add_argument('--use-confidence', action='store_true')
  parser.add_argument('--use-transformer', action='store_true')
  args = parser.parse_args()

  print('Reading src embeddings')
  src_tokenizer = Tokenizer(args.src_embeddings,
          bert_tokenization=args.bert_tokens)
  print('Reading mt embeddings')
  mt_tokenizer = Tokenizer(args.mt_embeddings,
          bert_tokenization=args.bert_tokens)

  print(f'Using \'{device}\' device.')

  collate = lambda data: qe_collate(data, device=device)

  train_ds = QEDataset('train', src_tokenizer, mt_tokenizer,
                       use_baseline=args.use_baseline,
                       use_bert=args.use_bert,
                       data_dir=args.train_path)
  train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size,
                            collate_fn=collate)

  dev_ds = QEDataset('dev', src_tokenizer, mt_tokenizer,
                     use_baseline=args.use_baseline,
                     use_bert=args.use_bert,
                     data_dir=args.dev_path)
  dev_loader = DataLoader(dev_ds, shuffle=True, batch_size=args.batch_size,
                          collate_fn=collate)


  if args.use_baseline:
    baseline_vocab_sizes = dev_ds._baseline_vocab_sizes
  else:
    baseline_vocab_sizes = None

  if args.use_bert:
    bert_features_size = 768 if args.bert_tokens else 2 * 768
  else:
    bert_features_size = 0

  model = QualityEstimator(args.hidden_size,
                           torch.tensor(src_tokenizer._embeddings),
                           torch.tensor(mt_tokenizer._embeddings),
                           baseline_vocab_sizes=baseline_vocab_sizes,
                           bert_features_size=bert_features_size,
                           transformer_encoder=args.use_transformer,
                           predict_gaps=args.predict_gaps).to(device)

  optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
  train(train_loader, dev_loader, model, optimizer, n_epochs=args.num_epochs,
        validate_every=args.validate_every, save_dir=args.checkpoint_dir)


if __name__ == '__main__':
  main()
