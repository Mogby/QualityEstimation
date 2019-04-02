#!/usr/bin/env python3

import argparse
import numpy as np
import pickle

from tqdm import tqdm


def feature_to_float(features, feature_idx):
  for sample_feature in features:
    for word_features in sample_feature:
      value = word_features[feature_idx]
      word_features[feature_idx] = np.asarray([value], dtype=np.float32)


def build_feature_vocab(features, feature_idx, occurences_threshold):
  print(f'Building vocab for feature {feature_idx}')
  num_occurences = {}

  for sample_features in features:
    for word_features in sample_features:
      value = word_features[feature_idx]
      if value not in num_occurences:
        num_occurences[value] = 0
      num_occurences[value] += 1

  vocab = [value for value, num in num_occurences.items()
           if num >= occurences_threshold]

  value2idx = {
    value: idx for idx, value in enumerate(vocab)
  }

  return value2idx


def feature_to_one_hot(features, feature_idx, value2idx):
  print(f'Converting {feature_idx} to one-hot')

  vocab_size = len(value2idx)

  for sample_features in tqdm(features):
    for word_features in sample_features:
      value = word_features[feature_idx]
      if value in value2idx:
        word_features[feature_idx] = np.asarray([value2idx[value]],
                                                dtype=np.float32)
      else:
        word_features[feature_idx] = np.asarray([vocab_size],
                                                dtype=np.float32)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--features-file', required=True,
                      type=argparse.FileType('r'))
  parser.add_argument('--features-out', required=True,
                      type=argparse.FileType('wb'))
  parser.add_argument('--occurences-threshold', default=4, type=int)
  parser.add_argument('--vocabs-file', type=argparse.FileType('rb'))
  parser.add_argument('--vocabs-out', type=argparse.FileType('wb'))
  args = parser.parse_args()

  print('Reading features')

  num_features = None
  features = []
  sample_features = []
  for line in args.features_file:
    if not line.strip():
      features.append(sample_features)
      sample_features = []
      continue

    sample_features.append(line.split())
    if num_features is not None:
      assert len(sample_features[-1]) == num_features
    else:
      num_features = len(sample_features[-1])

  if len(sample_features) > 0:
    features.append(sample_features)

  is_numerical = [True] * num_features
  for sample_features in features:
    for word_features in sample_features:
      for i, feature in enumerate(word_features):
        try:
          float(feature)
        except ValueError:
           is_numerical[i] = False

  print('Processing features')

  vocabs = [None] * num_features
  if args.vocabs_file is not None:
    vocabs = pickle.load(args.vocabs_file)
  else:
    for i in range(num_features):
      if not is_numerical[i]:
        vocabs[i] = build_feature_vocab(features, i, args.occurences_threshold)

  for i in range(num_features):
    if is_numerical[i]:
      feature_to_float(features, i)
    else:
      feature_to_one_hot(features, i, vocabs[i])

  for sample_features in features:
    for i in range(len(sample_features)):
      sample_features[i] = np.concatenate(sample_features[i])

  print('Dumping features')
  preprocessed = {
      'features': features,
      'vocab_sizes': [len(v) if v is not None else -1 for v in vocabs],
  }
  pickle.dump(preprocessed, args.features_out)

  if args.vocabs_out is not None:
    print('Dumping vocabs')
    pickle.dump(vocabs, args.vocabs_out)


if __name__ == '__main__':
  main()
