import pickle

import numpy as np


PAD_TOKEN = -1
UNK_TOKEN = -2


class Tokenizer:

  def __init__(self, embeddings_file):
    embeddings = pickle.load(embeddings_file)

    self._word2idx = embeddings['word2idx']
    self._idx2word = embeddings['idx2word'] + ['[UNK]'] + ['[PAD]']
    self._embeddings = np.asarray(embeddings['idx2vec'], dtype=np.float32)

  def tokenize(self, text):
    return text.split()

  def convert_tokens_to_ids(self, tokens):
    return list(map(
      lambda word: self._word2idx.get(word.lower(), UNK_TOKEN),
      tokens
    ))

  def tokens_to_sentence(self, tokens):
    return ' '.join(list(map(
      lambda token: self._idx2word[token],
      tokens
    )))

  def get_embeddings(self):
    return self._embeddings.copy()
