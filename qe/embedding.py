import pickle

import numpy as np

from pytorch_pretrained_bert import BertTokenizer


PAD_TOKEN = -1
UNK_TOKEN = -2


class Tokenizer:

  def __init__(self, embeddings_file=None, bert_tokenization=False):
    if embeddings_file is not None:
      self._has_embeds = True
      embeddings = pickle.load(embeddings_file)
      self._tokens_to_ids = embeddings['word2idx']
      self._ids_to_tokens =  embeddings['idx2word'] + ['[UNK]'] + ['[PAD]']
      self._embeddings = np.asarray(embeddings['idx2vec'], dtype=np.float32)
    else:
      self._has_embeds = False

    self._bert_tokenization = bert_tokenization
    if bert_tokenization:
      self._bert_tokenizer = \
        BertTokenizer.from_pretrained('bert-base-multilingual-cased')

  def tokenize(self, text):
    if self._bert_tokenization:
      return self._bert_tokenizer.tokenize(text)
    return text.split()

  def convert_tokens_to_ids(self, tokens):
    assert self._has_embeds

    return list(map(
      lambda word: self._tokens_to_ids.get(word, UNK_TOKEN),
      tokens
    ))

  def convert_ids_to_tokens(self, ids):
    assert self._has_embeds

    return list(map(
      lambda id: self._ids_to_tokens[id],
      ids
    ))

  def get_embeddings(self):
    assert self._has_embeds

    return self._embeddings.copy()
