#!/usr/bin/env python3

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_pretrained_bert import BertTokenizer
from sklearn.metrics import f1_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from qe.dataset import BertQEDataset
from qe.model import BertQE


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_score_on_dataset(model, dataloader):
  tokens_pred = np.empty(0)
  words_pred = np.empty(0)
  gaps_pred = np.empty(0)

  tokens_true = np.empty(0)
  words_true = np.empty(0)
  gaps_true = np.empty(0)

  with torch.no_grad():
    true_bad, true_ok, false_bad, false_ok = 0, 0, 0, 0
    for sample in tqdm(dataloader):
      tags = sample['tags']

      evaluation = model(sample['src'].to(device),
                         sample['segs'].to(device))

      mt_mask = sample['mt_mask'] == 1

      ok_tokens = evaluation[:, :, 1] > evaluation[:, :, 0]
      ok_tokens = ok_tokens[mt_mask]
      # ok_words = evaluation[0, 1::2, 1] > evaluation[0, 1::2, 0]
      # ok_gaps = evaluation[0, 0::2, 1] > evaluation[0, 0::2, 0]

      tokens_pred = np.hstack((tokens_pred, ok_tokens.cpu().numpy()))
      # words_pred = np.hstack((words_pred, ok_words.cpu().numpy()))
      # gaps_pred = np.hstack((gaps_pred, ok_gaps.cpu().numpy()))=

      tokens_true = np.hstack((tokens_true, sample['tags'][mt_mask]))
      # words_true = np.hstack((words_true, tags[1::2]))
      # gaps_true = np.hstack((gaps_true, tags[0::2]))

    preds = [tokens_pred]  # , words_pred, gaps_pred]
    truths = [tokens_true]  # , words_true, gaps_true]

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


def train_epochs(dataloader, model, n_epochs, validator,
                 validate_every=5, learning_rate=0.):
  def masked_nll_loss(input, target, mask):
    input_shape = input.shape
    loss = F.nll_loss(input.view(-1, input_shape[-1]),
                      target.view(-1),
                      reduction='none')
    loss = loss.view(input_shape[:-1])
    loss *= mask.to(torch.float)
    return loss.sum(dim=1).mean()

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  loss_hist = []
  for epoch in range(n_epochs):
    epoch_loss = 0
    print(f'Epoch {epoch+1}')
    for batch in tqdm(dataloader):
      pred = model(batch['src'].to(device),
                   batch['segs'].to(device))
      loss = masked_nll_loss(pred,
                             batch['tags'].to(device),
                             batch['mt_mask'].to(device))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f'avg_loss = {avg_loss}')
    loss_hist.append(avg_loss)

    if (epoch + 1) % validate_every == 0:
      validator(model)

    torch.save(model.state_dict(),
               f'drive/My Drive/colab/bert_{epoch}.torch')

  return loss_hist

def train_bert(dataloader, model, n_epochs, validator,
                 validate_every=5, learning_rate=0.):
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  loss_hist = []
  for epoch in range(n_epochs):
    epoch_loss = 0
    print(f'Epoch {epoch+1}')
    for batch in tqdm(dataloader):
      loss = model.loss(batch['src'].to(device), batch['segs'].to(device), batch['indices'].to(device), torch.tensor(batch['orig_tags']).to(device))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f'avg_loss = {avg_loss}')
    loss_hist.append(avg_loss)

    if (epoch + 1) % validate_every == 0:
      #validate
      pass

    torch.save(model.state_dict(),
               f'bert_tuned/bert_{epoch}.torch')

  return loss_hist

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('--train-path', required=True, type=str)
  parser.add_argument('--dev-path', required=True, type=str)
  parser.add_argument('--num-epochs', default=4, type=int)
  parser.add_argument('--learning-rate', default=2e-5, type=float)
  parser.add_argument('--batch-size', default=1, type=int)

  args = parser.parse_args()

  print('Using device:', device)

  tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
  # dataset = BertQEDataset('train', args.train_path, tokenizer)
  dev_dataset = BertQEDataset('dev', args.dev_path, tokenizer)
  model = BertQE().to(device)

  # dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
  devloader = DataLoader(dev_dataset, args.batch_size, shuffle=False)
  # validator = lambda model: print(get_score_on_dataset(model, dataloader),
  #                                 get_score_on_dataset(model, devloader),
  #                                 sep='\n')

  loss = train_bert(devloader, model, args.num_epochs, validator=None,
                      validate_every=1, learning_rate=args.learning_rate)


if __name__ == '__main__':
  main()