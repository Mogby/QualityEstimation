#!/usr/bin/env python3

import argparse
import pickle

from gensim.models import FastText


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fasttext-model', metavar='<model>', required=True,
                      type=str)
  parser.add_argument('--output-file', metavar='<output>', required=True,
                      type=argparse.FileType('wb'))
  parser.add_argument('files', metavar='<textfile>', nargs='+')

  args = parser.parse_args()

  print('Loading model...')
  ft = FastText.load_fasttext_format(args.fasttext_model)

  print('Building vocab...', end=' ')
  vocab = []
  for fname in args.files:
    with open(fname, 'r') as f:
      for line in f:
        vocab.extend(line.split())
  vocab = list(set(vocab))  # remove duplicates
  vocab = sorted(vocab)
  word2idx = {word: i for i, word in enumerate(vocab)}
  print(f'vocab size is {len(vocab)}.')

  print('Precalculating embeddings...', end=' ')
  vecs = []
  num_unknown = 0
  for word in vocab:
    try:
      vecs.append(ft.wv[word])
    except:
      vecs.append(ft.wv['_'])
      num_unknown += 1

  print(f'encountered {num_unknown} unknown words.')

  print('Saving embeddings...')
  pickle.dump({
    'word2idx': word2idx,
    'idx2word': vocab,
    'idx2vec': vecs,
  }, args.output_file)

  print('Done.')


if __name__ == '__main__':
  main()
