#!/usr/bin/env python3

import argparse
import pickle

from gensim.models import FastText

from qe.embedding import Tokenizer

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fasttext-model', metavar='<model>', required=True,
                      type=str)
  parser.add_argument('--output-file', metavar='<output>', required=True,
                      type=argparse.FileType('wb'))
  parser.add_argument('--bert-tokens', action='store_true')
  parser.add_argument('files', metavar='<textfile>', nargs='+')

  args = parser.parse_args()

  if args.bert_tokens:
    print('Using bert tokens...')

  print('Loading model...')
  ft = FastText.load_fasttext_format(args.fasttext_model)

  print('Building vocab...', end=' ')
  tokenizer = Tokenizer(bert_tokenization=args.bert_tokens)
  vocab = []
  for fname in args.files:
    print(f'Reading {fname}...')
    with open(fname, 'r', encoding='utf-8') as f:
      for line in f:
        for word in line.split():
          vocab.extend(tokenizer.tokenize(word))
  vocab = list(set(vocab))  # remove duplicates
  vocab = sorted(vocab)
  word2idx = {word: i for i, word in enumerate(vocab)}
  print(f'Vocab size is {len(vocab)}...')

  print('Precalculating embeddings...')
  vecs = []
  for word in vocab:
    try:
      vecs.append(ft.wv[word])
    except:
      print('Unknown word:', word)
      vecs.append(ft.wv['_'])

  print('Saving embeddings...')
  pickle.dump({
    'word2idx': word2idx,
    'idx2word': vocab,
    'idx2vec': vecs,
  }, args.output_file)

  print('Done.')


if __name__ == '__main__':
  main()
