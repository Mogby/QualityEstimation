#!/usr/bin/env python3

import argparse

import torch

from torch.utils.data import random_split, DataLoader

from qe.dataset import QEDataset, qe_collate
from qe.embedding import Tokenizer
from qe.model import device, QualityEstimator
from qe.train import validate


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset-path', required=True, type=str)
  parser.add_argument('--dataset-name', required=True, type=str)
  parser.add_argument('--src-embeddings', required=True,
                      type=argparse.FileType('rb'))
  parser.add_argument('--mt-embeddings', required=True,
                      type=argparse.FileType('rb'))
  parser.add_argument('--model-file', required=True, type=str)
  args = parser.parse_args()

  print('Reading src embeddings')
  src_tokenizer = Tokenizer(args.src_embeddings)
  print('Reading mt embeddings')
  mt_tokenizer = Tokenizer(args.mt_embeddings)

  print(f'Using \'{device}\' device.')

  data = QEDataset(args.dataset_name, src_tokenizer, mt_tokenizer,
                   data_dir=args.dataset_path, use_bert=True, use_baseline=True)
  collate = lambda data: qe_collate(data, device=device)
  data_loader = DataLoader(data, collate_fn=collate)

  model = QualityEstimator(900,
                           torch.tensor(src_tokenizer._embeddings),
                           torch.tensor(mt_tokenizer._embeddings),
                           baseline_vocab_sizes=data._baseline_vocab_sizes,
                           bert_features_size=768*2,
                           transformer_encoder=False,
                           predict_gaps=True,
                           ).to(device)
  model.load_state_dict(torch.load(args.model_file, map_location='cpu'))
  model.eval()

  # print(validate(data_loader, model))
  # return

  with open('tags', 'w') as fmt:
    with open('source_tags', 'w') as fsrc:
      for i, item in enumerate(data_loader):
        kwargs = {}
        if 'bert_features' in item:
          kwargs['bert_features'] = item['bert_features']
        if 'baseline_features' in item:
          kwargs['baseline_features'] = item['baseline_features']

        src_tag, mt_tag, gap_tag = model.predict(
            item['src'],
            item['mt'],
            item['aligns'],
            **kwargs
        )

        for t in src_tag[:,0]:
          print('BAD' if t == 0 else 'OK', end=' ', file=fsrc)
        print('', file=fsrc)
        tags_merged = torch.zeros(len(mt_tag) + len(gap_tag))
        tags_merged[::2] = gap_tag[:,0]
        tags_merged[1::2] = mt_tag[:,0]
        for t in tags_merged:
          print('BAD' if t == 0 else 'OK', end=' ', file=fmt)
        print('', file=fmt)


if __name__ == '__main__':
  main()
