import numpy as np


UNK_IDX = -1


class Tokenizer:

  def __init__(self, glove_file):
    self._word2idx = {}
    self._idx2word = []
    self._embeddings = []

    for i, line in enumerate(glove_file):
      tokens = line.split()
      word = tokens[0]
      embedding = [float(coord) for coord in tokens[1:]]
      self._word2idx[word] = i
      self._idx2word.append(word)
      self._embeddings.append(embedding)

    self._idx2word.append('[UNK]')

    self._embeddings = np.array(self._embeddings, dtype=np.float32)

  def tokenize_sentence(self, sentence):
    return list(map(
      lambda word: self._word2idx.get(word.lower(), UNK_IDX),
      sentence.split()
    ))

  def tokens_to_sentence(self, tokens):
    return ' '.join(list(map(
      lambda token: self._idx2word[token],
      tokens
    )))

  def get_embeddings(self):
    return self._embeddings.copy()
