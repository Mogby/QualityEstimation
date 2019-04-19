#!/usr/bin/env python3

import argparse
import pickle

import numpy as np
import torch

from pytorch_pretrained_bert import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from qe.dataset import BertQEDataset
from qe.model import device

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset-path', required=True, type=str)
  parser.add_argument('--dataset-name', required=True, type=str)
  parser.add_argument('--out-file', required=True, type=argparse.FileType('wb'))
  parser.add_argument('--bert-tokens', action='store_true')
  args = parser.parse_args()

  if args.bert_tokens:
    print('Using bert tokens...')

  model_name = 'bert-base-multilingual-cased'
  tokenizer = BertTokenizer.from_pretrained(model_name)
  bert = BertModel.from_pretrained(model_name).to(device)

  dataset = BertQEDataset(args.dataset_name, args.dataset_path, tokenizer)
  dataloader = DataLoader(dataset)

  print('Generating features...')
  with torch.no_grad():
    features = []
    for sample in tqdm(dataloader):
      indices = sample['indices'].to(device)
      segs = sample['segs'].to(torch.uint8).to(device)

      bert_out, _ = bert(sample['src'].to(device), sample['segs'].to(device),
                         output_all_encoded_layers=False)

      mt_indices = indices[segs][:-1]
      bert_mt_features = bert_out[segs][:-1]

      sample_features = []
      max_idx = mt_indices.max()
      if args.bert_tokens:
        sample_features = bert_mt_features.cpu().numpy()
      else:
        for i in range(max_idx + 1):
          idx_mask = mt_indices == i
          sample_features.append(torch.cat((
            bert_mt_features[idx_mask][0],
            bert_mt_features[idx_mask][-1],
          )).cpu().numpy())

      features.append(np.asarray(sample_features, dtype=np.float32))

  print('Dumping features')
  pickle.dump(features, args.out_file)


if __name__ == '__main__':
  main()
