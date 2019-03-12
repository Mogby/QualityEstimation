import os

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

def train_epochs(dataloader, estimator, n_epochs, validator=None,
                 validate_every=5, batch_size=100, learning_rate=0.1,
                 save_dir=None):
  optimizer = optim.Adadelta(estimator.parameters(), lr=learning_rate)

  criterion = nn.NLLLoss()

  loss_hist = []
  for epoch in range(n_epochs):
    print(f'Epoch {epoch+1}')
    loss = 0
    epoch_loss = 0
    for iter, sample in enumerate(tqdm(dataloader)):
      src, mt, tags = sample['src'], sample['mt'], sample['tags']

      pred, _, _ = estimator(src[0], mt[0], training=True)
      iter_loss = criterion(pred[0], tags[0])

      loss += iter_loss

      if (iter + 1) % batch_size == 0 or iter == len(dataloader) - 1:
        loss /= batch_size
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = 0

    epoch_loss_avg = epoch_loss / len(dataloader)
    print(f'avg_loss = {epoch_loss_avg}')
    loss_hist.append(epoch_loss_avg)

    if validator is not None and (epoch + 1) % validate_every == 0:
      validator(estimator)

    if save_dir is not None:
      model_file = f'model_{epoch + 1}.torch'
      torch.save(estimator.state_dict(),
                 os.path.join(save_dir, model_file))

  return loss_hist