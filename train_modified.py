import future
import copy
import logging
import os
import sys
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import utilities as utils
from tqdm import tqdm

from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold

from datasets import CovidDataset
import networks

torch.cuda.current_device()

# ------------------------------------------------------------------------------------
#                                       Some Paths to be used
# ------------------------------------------------------------------------------------
dir_cwd = os.getcwd()
dir_data_root = os.path.join(dir_cwd, 'data', 'covid_data')
dir_checkpoint = os.path.join(dir_cwd, 'checkpoints')
correct_labels = os.path.join(dir_cwd, 'labels.csv')

# Random Seed TO Produce Same Data Distributions
seed_val = 131


# ------------------------------------------------------------------------------------
#                           Perform Stratified KFold Cross Validation
# ------------------------------------------------------------------------------------

def perform_skfcv(model, device, inp_df, folds=5, epochs=10, batch_size=8, lr=0.1, save_cp=False, optim='adam',):
    logging.info(f'''Performing {folds}folds Cross Validation''')
    # store train/val history
    train_losses = []
    train_accuracies = []
    train_metrics = []
    val_losses = []
    val_accuracies = []
    val_metrics = []
    skf = StratifiedKFold(n_splits=folds)
    x = np.array(inp_df.x)
    y = np.array(inp_df.y)
    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tr = {
            'x': X_train,
            'y': y_train
        }
        ts = {
            'x': X_test,
            'y': y_test
        }
        tr_df = pd.DataFrame(tr)
        ts_df = pd.DataFrame(ts)
        train_epoch_loss, train_epoch_acc, train_epoch_metrics, val_epoch_loss, val_epoch_acc, val_epoch_metrics = train_net(
            model=model,
            device=device,
            train_df=tr_df,
            val_df=ts_df,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_cp=save_cp,
            optim=optim
        )
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_acc)
        train_metrics.append(train_epoch_metrics)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        val_metrics.append(val_epoch_metrics)

    return train_losses, train_accuracies, train_metrics, val_losses, val_accuracies, val_metrics


# ------------------------------------------------------------------------------------
#                                       Training the Model
# ------------------------------------------------------------------------------------
def train_net(
        model,
        device,
        train_df,
        val_df,
        epochs=10,
        batch_size=8,
        lr=0.1,
        save_cp=True,
        optim='adam',
        cv_runs=''
):
    resize = (128, 128)

    # Now, let's build dataset objects
    # Building up the Dataset objects
    train_set = CovidDataset(
        root=dir_data_root,
        inp_df=train_df,
        transformations=utils.get_transformations(for_train=True)
    )

    val_set = CovidDataset(
        root=dir_data_root,
        inp_df=val_df,
        transformations=utils.get_transformations(for_train=False)
    )

    # Creating DataLoaders for each set
    train_set_dl = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    val_set_dl = DataLoader(
        val_set,
        batch_size=4,
        shuffle=True
    )

    # dictionary of dataloaders
    dataloaders = {'train': train_set_dl,
                   'val': val_set_dl,
                   }

    # Deciding the optimizer
    if optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay=0.001, nesterov=True)

    # Setting up loss based
    criterion = nn.CrossEntropyLoss()

    params = {
        'optimizer': optimizer,
        'criterion': criterion
    }

    optim_string = optimizer.__str__().replace("\n", ' ')

    logging.info(f'''Starting training:
        Input Type:         Chest XRays - RGB Images
        Output Type;        0/1 - Classification
        Batch Norm:         True
        Activation:         ReLU
        Epochs:             {epochs}
        Batch size:         {batch_size}
        Optimizer:          {optim_string}
        Learning rate:      {lr}
        Training size:      {train_set.__len__()}
        Validation size:    {val_set.__len__()}
        Checkpoints:        {save_cp}
        Device:             {device.type}
        Image Size:         {resize}
    ''')

    # store train/val loss history
    train_epoch_loss = []
    train_epoch_acc = []
    train_epoch_metrics = []
    val_epoch_loss = []
    val_epoch_acc = []
    val_epoch_metrics = []

    # To create reference for making decision for best results
    prev_val_loss = np.Infinity  # Any arbitrary Number would do fine
    prev_val_acc = 0.0
    prev_val_f1 = 0.0

    # Initialization to save best weights and model
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):

        start_time = time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
                set_len = len(train_set)
                desc = f'Epoch {epoch + 1}/{epochs}'
                leave = True
            else:
                model.eval()  # Set model to evaluate mode i.e. freeze weight updates
                set_len = len(val_set)
                desc = 'Validation Phase'
                leave = True

            total = 0
            running_loss = 0
            correct_preds = 0
            original_labels = []
            predicted_labels = []

            with tqdm(total=set_len, desc=desc, unit='img', leave=leave) as pbar:
                for batch in dataloaders[phase]:  # Get Each Batch According to Phase
                    _, images, labels = batch  # Let's get images and labels for each batch

                    images = images.to(device)
                    labels = labels.to(device)

                    # We want to zero out the gradients every time as by default pytorch accumulates gradients
                    optimizer.zero_grad()

                    # We want to calculate and update gradients only in 'train phase'
                    if phase == 'val':
                        with torch.no_grad():
                            outputs = model(images)
                            loss = params['criterion'](outputs, labels).cuda()

                            pbar.set_postfix(**{'Val CE loss (running)': loss.item()})

                            total += images.size(0)
                            _, predicted = torch.max(outputs.data, 1)
                            correct_preds += (predicted == labels).sum().item()

                            original_labels += labels.data.cpu().numpy().tolist()
                            predicted_labels += predicted.data.cpu().numpy().tolist()
                            pbar.update(images.shape[0])

                    else:
                        # forward pass
                        outputs = model(images)
                        loss = params['criterion'](outputs, labels).cuda()

                        pbar.set_postfix(**{'train CE Loss (running)': loss.item()})

                        loss.backward()  # Calculate Gradients
                        params['optimizer'].step()  # Update Weights

                        total += images.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        correct_preds += (predicted == labels).sum().item()

                        original_labels += labels.data.cpu().numpy().tolist()
                        predicted_labels += predicted.data.cpu().numpy().tolist()

                        pbar.update(images.shape[0])

                    running_loss += loss.item() * images.size(0)

                epoch_loss = running_loss / total
                epoch_acc = (correct_preds / total) * 100
                epoch_metrics = utils.evaluate_metrics(original_labels, predicted_labels)

                if phase == 'train':
                    train_epoch_loss.append(epoch_loss)
                    train_epoch_acc.append(epoch_acc)
                    logging.info(f'''Train:
                        Cross Entropy Loss:     {epoch_loss}
                        Epoch Accuracy:         {epoch_acc}
                        F1-Score:               {epoch_metrics['f1_score']} 
                        Precision:              {epoch_metrics['precision']}
                        Recall:                 {epoch_metrics['recall']}
                        Specificity:            {epoch_metrics['specificity']}
                        Sensitivity:            {epoch_metrics['sensitivity']}
                    ''')
                elif phase == 'val':

                    logging.info(f'''Validation:
                                            Cross Entropy Loss:     {epoch_loss}
                                            Epoch Accuracy:         {epoch_acc}
                                            F1-Score:               {epoch_metrics['f1_score']} 
                                            Precision:              {epoch_metrics['precision']}
                                            Recall:                 {epoch_metrics['recall']}
                                            Specificity:            {epoch_metrics['specificity']}
                                            Sensitivity:            {epoch_metrics['sensitivity']}
                                        ''')
                    val_epoch_loss.append(epoch_loss)
                    val_epoch_acc.append(epoch_acc)
                    if round(epoch_loss, 5) < prev_val_loss and round(epoch_acc, 5) > prev_val_acc and round(
                            epoch_metrics['f1_score'], 5) > prev_val_f1:
                        prev_val_loss = epoch_loss
                        prev_val_acc = epoch_acc
                        prev_val_f1 = epoch_metrics['f1_score']
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_res = f'''Best Results:
                                                    Val Loss:   {epoch_loss}
                                                    Accuracy:   {epoch_acc}
                                                    Metrics:    {epoch_metrics}
                                                '''
                        logging.info(best_res)

        if save_cp:
            try:
                if not os.path.exists(dir_checkpoint):
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
            except OSError:
                pass
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(),
                           os.path.join(dir_checkpoint, f'{model.name}{epoch + 1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved !')

        end_time = time()
        logging.info('Epoch took time: {}'.format(str(end_time - start_time)))

    # Load n Save Best Model Weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(dir_checkpoint, f'{model.name}_{cv_runs}_best_weights.pth'))

    return train_epoch_loss, train_epoch_acc, train_epoch_metrics, val_epoch_loss, val_epoch_acc, val_epoch_metrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # empty cuda cache
    torch.cuda.empty_cache()
    logging.info(f'Using device {device}')

    model = networks.CustomVGG16()

    # We may Load state from checkpoints if something goes wrong while training
    # ToDo Code Implementation 'Start Training from last checkpoint'

    model.to(device=device)

    # Arguments
    num_epochs = 20
    batch_size = 8
    learning_rate = 0.0001

    # preparing for train and val sets
    # First we read 'labels.csv' and construct input dataframe
    inp_df = pd.read_csv(correct_labels)

    # Now we get the stratified DataFrames for input to our Dataset Objects
    train_df, val_df, _ = utils.get_stratified_train_val_test_sets(inp_df=inp_df, seed=seed_val)
    try:
        train_epoch_loss, train_epoch_acc, train_epoch_metrics, val_epoch_loss, val_epoch_acc, val_epoch_metrics = train_net(
            model=model,
            device=device,
            train_df=train_df,
            val_df=val_df,
            epochs=num_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            save_cp=False,
            optim='sgd'
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(dir_checkpoint, f'{model.name}_INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
