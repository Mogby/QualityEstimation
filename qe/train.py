import os

import numpy as np
import torch

from tqdm import tqdm
from sklearn.metrics import f1_score


def validate(val_loader, model):
  with torch.no_grad():
    src_preds = []
    word_preds = []
    gap_preds = []

    src_truths = []
    word_truths = []
    gap_truths = []

    for batch in tqdm(val_loader):
      src_mask = batch['src_tags'] >= 0
      word_mask = batch['word_tags'] >= 0
      gap_mask = batch['gap_tags'] >= 0

      kwargs = {
          'src': batch['src'],
          'mt': batch['mt'],
      }
      if 'bert_features' in batch:
        kwargs['bert_features'] = batch['bert_features']
      if 'baseline_features' in batch:
        kwargs['baseline_features'] = batch['baseline_features']
      src_pred, mt_pred = model.predict(**kwargs)

      src_preds.append(src_pred[src_mask].cpu().numpy())
      word_preds.append(mt_pred[1::2][word_mask].cpu().numpy())
      gap_preds.append(mt_pred[::2][gap_mask].cpu().numpy())

      src_truths.append(batch['src_tags'][src_mask].cpu().numpy())
      word_truths.append(batch['word_tags'][word_mask].cpu().numpy())
      gap_truths.append(batch['gap_tags'][gap_mask].cpu().numpy())

    scores = {
      'src': (src_preds, src_truths),
      'word': (word_preds, word_truths),
      'gap': (gap_preds, gap_truths),
    }

    for token, (preds, truths) in scores.items():
      f1_bad, f1_ok = f1_score(np.concatenate(preds),
                               np.concatenate(truths),
                               average=None)
      scores[token] = {
        'f1_ok': f1_ok,
        'f1_bad': f1_bad,
        'mul': f1_ok * f1_bad,
      }

    return scores


def train(train_loader, val_loader, model, optimizer, n_epochs,
          validate_every=5, save_dir=None):
  loss_hist = []
  for epoch in range(n_epochs):
    print(f'Epoch {epoch+1}')
    epoch_loss = 0
    for batch in tqdm(train_loader):
      loss = model.loss(**batch)
      epoch_loss += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    epoch_loss_avg = epoch_loss / len(train_loader)
    print(f'avg_loss = {epoch_loss_avg}')
    loss_hist.append(epoch_loss_avg)

    if (epoch + 1) % validate_every == 0:
      print('Validating')
      scores = validate(val_loader, model)
      print('scores =', scores)

    if save_dir is not None:
      save_dict = {
        f'model_{epoch+1}.pth': model,
        f'optim_{epoch+1}.pth': optimizer,
      }

      for filename, object in save_dict.items():
        torch.save(object.state_dict(),
                   os.path.join(save_dir, filename))

  return loss_hist
