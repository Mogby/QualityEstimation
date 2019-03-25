#!/usr/bin/env python3

import argparse
import os
import pickle

import numpy as np
import torch

from sklearn.metrics import f1_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from qe.dataset import QEDataset
from qe.model import EstimatorCRF


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_crf(dataloader, crf, n_epochs, validator,
              validate_every=5, batch_size=5, learning_rate=0.1,
              model_checkpoint=None, optim_checkpoint=None):
  print_loss_total = 0

  optimizer = optim.Adadelta(crf.parameters(), lr=learning_rate)

  if model_checkpoint is not None:
    crf.load_state_dict(torch.load(model_checkpoint))
  if optim_checkpoint is not None:
    optimizer.load_state_dict(torch.load(optim_checkpoint))

  loss = 0
  loss_hist = []

  for epoch in range(n_epochs):
    print(f'Epoch {epoch+1}')
    for iter, sample in enumerate(tqdm(dataloader)):
      src, mt, tags = sample['src'], sample['mt'], sample['tags']

      iter_loss = -crf.log_likelihood(src, mt, tags[0],
                                      training=True)

      loss += iter_loss
      print_loss_total += iter_loss.item()

      if (iter + 1) % batch_size == 0 or iter == len(dataloader) - 1:
        loss /= batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = 0

    print_loss_avg = print_loss_total / len(dataloader)
    print_loss_total = 0
    print(f'avg_loss = {print_loss_avg}')
    loss_hist.append(print_loss_avg)

    if (epoch + 1) % validate_every == 0:
      validator(crf)

    torch.save(optimizer.state_dict(),
               f'crf_optim_{epoch}.torch')
    torch.save(crf.state_dict(),
               f'crf_{epoch}.torch')

  return loss_hist


def get_crf_score_on_dataset(crf, dataset):
  tokens_pred = np.empty(0)
  words_pred = np.empty(0)
  gaps_pred = np.empty(0)

  tokens_true = np.empty(0)
  words_true = np.empty(0)
  gaps_true = np.empty(0)

  with torch.no_grad():
    for sample in tqdm(dataset):
      src = sample['src'].unsqueeze(0)
      mt = sample['mt'].unsqueeze(0)
      tags = sample['tags'].cpu().numpy()

      evaluation, _ = crf.label(src, mt)

      ok_tokens = evaluation[:]
      ok_words = evaluation[1::2]
      ok_gaps = evaluation[0::2]

      tokens_pred = np.hstack((tokens_pred, ok_tokens.cpu().numpy()))
      words_pred = np.hstack((words_pred, ok_words.cpu().numpy()))
      gaps_pred = np.hstack((gaps_pred, ok_gaps.cpu().numpy()))

      tokens_true = np.hstack((tokens_true, tags))
      words_true = np.hstack((words_true, tags[1::2]))
      gaps_true = np.hstack((gaps_true, tags[0::2]))

    preds = [tokens_pred, words_pred, gaps_pred]
    truths = [tokens_true, words_true, gaps_true]

    scores = []
    for pred, true in zip(preds, truths):
      f1_ok = f1_score(true, pred)
      f1_bad = f1_score(1 - true, 1 - pred)
      scores.append([
        f1_ok,
        f1_bad,
        f1_ok * f1_bad
      ])

    return scores


def validate_crf(crf, train, dev):
  train_score = get_crf_score_on_dataset(crf, train)
  dev_score = get_crf_score_on_dataset(crf, dev)

  print('Train score:')
  print(train_score)
  print('Dev score:')
  print(dev_score)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--src-embeddings', required=True,
                      type=argparse.FileType('r', encoding='utf8'))
  parser.add_argument('--mt-embeddings', required=True,
                      type=argparse.FileType('r', encoding='utf8'))
  parser.add_argument('--train-path', required=True,
                      type=str)
  parser.add_argument('--dev-path', required=True,
                      type=str)
  parser.add_argument('--num-epochs', default=20, type=int)
  args = parser.parse_args()

  print('Reading src embeddings')
  src_emb = pickle.load(args.src_embeddings)
  src_wv = {word:src_emb['idx2vec'][idx]
            for idx, word in enumerate(src_emb['idx2word'])}

  print('Reading mt embeddings')
  mt_emb = pickle.load(args.mt_embeddings)
  mt_wv = {word: mt_emb['idx2vec'][idx]
           for idx, word in enumerate(mt_emb['idx2word'])}

  dataset = QEDataset('train', src_wv, mt_wv, spoil_p=0.,
                      data_dir=args.train_path)
  dev = QEDataset('dev', src_wv, mt_wv, spoil_p=0.,
                  data_dir=args.dev_path)

  loader = DataLoader(dataset, shuffle=True)

  validator = lambda crf: validate_crf(crf, dataset, dev)

  estimator = EstimatorCRF(300, 100,
                           dropout_p=.0,
                           self_attn=True).to(device)

  train_crf(loader, estimator, args.num_epochs,
            validator=validator, evaluate_every=1,
            learning_rate=1)



if __name__ == '__main__':
  main()