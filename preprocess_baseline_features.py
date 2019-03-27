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


def feature_to_one_hot(features, feature_idx, occurences_threshold):
  print(f'Converting {feature_idx} to one-hot')
  num_occurences = {}

  for sample_features in features:
    for word_features in sample_features:
      value = word_features[feature_idx]
      if value not in num_occurences:
        num_occurences[value] = 0
      num_occurences[value] += 1

  vocab = [value for value, num in num_occurences.items()
                 if num >= occurences_threshold]
  vocab_size = len(vocab)

  value2idx = {
    value: idx for idx, value in enumerate(vocab)
  }

  for sample_features in tqdm(features):
    for word_features in sample_features:
      value = word_features[feature_idx]
      if value in vocab:
        word_features[feature_idx] = np.asarray([value2idx[value]],
                                                dtype=np.float32)
      else:
        word_features[feature_idx] = np.asarray([vocab_size],
                                                dtype=np.float32)

  return vocab_size


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--features-file', required=True,
                      type=argparse.FileType('r'))
  parser.add_argument('--out-file', required=True, type=argparse.FileType('wb'))
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

  vocab_size = [-1] * num_features
  for i in range(num_features):
    if is_numerical[i]:
      feature_to_float(features, i)
    else:
      vocab_size[i] = feature_to_one_hot(features, i, occurences_threshold=4)

  for sample_features in features:
    for i in range(len(sample_features)):
      sample_features[i] = np.concatenate(sample_features[i])

  print('Dumping features')

  preprocessed = {
      'features': features,
      'vocab_size': vocab_size,
  }

  pickle.dump(preprocessed, args.out_file)


if __name__ == '__main__':
  main()
