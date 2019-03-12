import argparse
import os
import pickle

import numpy as np
import torch

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from qe.dataset import  QEDataset
from qe.embedding import Tokenizer
from qe.model import EstimatorRNN
from qe.train import train_epochs


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_score_on_dataset(estimator, dataset):
  tokens_pred = np.empty(0)
  words_pred = np.empty(0)
  gaps_pred = np.empty(0)

  tokens_true = np.empty(0)
  words_true = np.empty(0)
  gaps_true = np.empty(0)

  with torch.no_grad():
    for sample in tqdm(dataset):
      src = sample['src']
      mt = sample['mt']
      tags = sample['tags'].cpu().numpy()

      evaluation, _, _ = estimator(src, mt)

      ok_tokens = evaluation[0, :, 1] > evaluation[0, :, 0]
      ok_words = evaluation[0, 1::2, 1] > evaluation[0, 1::2, 0]
      ok_gaps = evaluation[0, 0::2, 1] > evaluation[0, 0::2, 0]

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


def validator(estimator, train_ds, dev_ds):
  train_score = get_score_on_dataset(estimator, train_ds)
  dev_score = get_score_on_dataset(estimator, dev_ds)

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
  args = parser.parse_args()

  print('Reading src embeddings')
  src_tokenizer = Tokenizer(args.src_embeddings)
  print('Reading mt embeddings')
  # mt_tokenizer = Tokenizer(args.mt_embeddings)

  tokens = src_tokenizer.tokenize_sentence('Hi I\'m Mark')
  print(tokens)
  print(src_tokenizer.tokens_to_sentence(tokens))
  return
  prefix = 'data'
  vec_files = ['de_vecs_compressed.pickle',
               'en_vecs_compressed.pickle']
  ft = {}

  for lang, file in zip(['de', 'en'], vec_files):
    path = os.path.join(prefix, file)
    with open(path, 'rb') as fin:
      ft[lang] = pickle.load(fin)
      print('Loaded', path)

  dataset = QEDataset('train', ft['en'], ft['de'], spoil_p=0.,
                      data_dir='data\\train\\en_de.nmt')
  dev = QEDataset('dev', ft['en'], ft['de'], spoil_p=0.,
                  data_dir='.\\data\\train\\en_de.nmt')
  loader = DataLoader(dataset, shuffle=True)

  validator_scores = []

  n_epochs = 20

  for self_attention in [True, False]:
    for dropout_p in [.0, .1, .2]:
      print('self_attention =', self_attention)
      print('dropout_p =', dropout_p)
      estimator = EstimatorRNN(150, 100,
                               dropout_p=dropout_p,
                               self_attn=self_attention).to(device)

      validator_scores = []
      losses = train_epochs(loader, estimator, n_epochs,
                            validator=validator, validate_every=4,
                            learning_rate=1)

      print('losses =', losses)
      print('validator_scores =', validator_scores)


if __name__ == '__main__':
  main()