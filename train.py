import copy
import logging
import os
import sys
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

import utilities as utils
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets import CovidDataset
import networks

# ------------------------------------------------------------------------------------
#                                       Some Paths to be used
# ------------------------------------------------------------------------------------
dir_cwd = os.getcwd()
dir_data_root = os.path.join(dir_cwd, 'data', 'covid_data')
dir_checkpoint = os.path.join(dir_cwd, 'checkpoints')
correct_labels = os.path.join(dir_cwd, 'labels.csv')


# ------------------------------------------------------------------------------------
#                                       Metrics Evaluation
# ------------------------------------------------------------------------------------
def evaluate_metrics(y_true, y_preds):
    """
    :param y_true: array
    :param y_preds: array

    Computes:
        Average Precision
        F1 Score
        Recall
        Precision
        Sensitivity
        Specificity
        Accuracy

    *Note:
        Problem is of binary classification

    :return: metrics: dictionary
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    metrics = dict()
    metrics['f1_score'] = f1_score(y_true, y_preds, pos_label=1)
    metrics['recall'] = recall_score(y_true, y_preds)
    metrics['precision'] = precision_score(y_true, y_preds)
    metrics['specificity'] = specificity
    metrics['sensitivity'] = sensitivity

    return metrics


# ------------------------------------------------------------------------------------
#                                       Training the Model
# ------------------------------------------------------------------------------------
def train_net(model, device, epochs=10, batch_size=8, lr=0.1, save_cp=True, optim='adam'):
    # First we read 'correct_labels.csv' and construct input dataframe
    inp_df = pd.read_csv(correct_labels)

    # Now we get the stratified DataFrames for input to our Dataset Objects
    train_df, val_df, test_df = utils.get_stratified_train_val_test_sets(inp_df=inp_df)

    resize = (128, 128)
    # Let's prepare the transformations to be applied on our datasets
    transformations = {
        'train_transforms': transforms.Compose([
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.RandomAffine(degrees=0, shear=(-0.05, 0.05)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(resize),
            transforms.ToTensor()]),
        'test_transforms': transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()])
    }

    # Now, let's build dataset objects
    # Building up the Dataset objects
    train_set = CovidDataset(
        root=dir_data_root,
        inp_df=train_df,
        transformations=transformations['train_transforms']
    )

    val_set = CovidDataset(
        root=dir_data_root,
        inp_df=val_df,
        transformations=transformations['test_transforms']
    )

    test_set = CovidDataset(
        root=dir_data_root,
        inp_df=test_df,
        transformations=transformations['test_transforms']
    )

    # Creating DataLoaders for each set
    train_set_dl = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True
    )

    val_set_dl = DataLoader(
        val_set,
        batch_size=4,
        shuffle=True
    )

    test_set_dl = DataLoader(
        test_set,
        batch_size=4,
        shuffle=True
    )

    # dictionary of dataloaders
    dataloaders = {'train': train_set_dl,
                   'val': val_set_dl,
                   'test': test_set_dl}

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

    # Let's create a tensorboard object
    writer = SummaryWriter(log_dir='runs/train_proposed_model_run1',
                           comment=f'OPTIM_{optim}_LOSS_{criterion}_LR_{lr}_BATCH_SIZE_{batch_size}_IMG_SIZE_{resize}')
    # Let's create a random input which may be passed to network and its depiction be displayed in tensorboard graph viz
    ran_inp = torch.randn((2, 3, 128, 128), device=device)
    writer.add_graph(model=model, input_to_model=ran_inp)
    optim_string = optimizer.__str__().replace("\n", ' ')
    text = f'''
        Input Type:         Chest XRays - RGB Images
        Output Type;        0/1 - Classification
        Batch Norm:         True
        Activation:         ReLU
        Epochs:             {epochs}
        Optimizer:          {optim_string}  
        Learning Rate:      {lr} 
        Train Batch Size:   {batch_size}
        Val Batch Size:     4 
        Loss Criterion:     {criterion.__repr__()}  
        Weight Init:        Default 
        '''
    writer.add_text('Configurations', text, 1)
    writer.flush()

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Optimizer:       {optim}
        Learning rate:   {lr}
        Training size:   {train_set.__len__()}
        Validation size: {val_set.__len__()}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Image Size:      {resize}
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
                epoch_metrics = evaluate_metrics(original_labels, predicted_labels)

                if phase == 'train':
                    train_epoch_loss.append(epoch_loss)
                    train_epoch_acc.append(epoch_acc)
                    # logging.info(f'''Train:
                    #     CE:    {epoch_loss}
                    # ''')
                    writer.add_scalar('Loss/Train/Cross Entropy Loss', epoch_loss, (epoch + 1))
                    writer.add_scalar('Metrics/Train/Accuracy', epoch_acc, (epoch + 1))
                    writer.add_pr_curve('Metrics/Train/PR-Curve', np.asarray(original_labels), np.asarray(predicted_labels), (epoch + 1))
                    writer.add_scalar('Metrics/Train/F1-Score', epoch_metrics['f1_score'], (epoch + 1))
                    writer.add_scalar('Metrics/Train/Precision', epoch_metrics['precision'], (epoch + 1))
                    writer.add_scalar('Metrics/Train/Recall', epoch_metrics['recall'], (epoch + 1))
                    writer.add_scalar('Metrics/Train/Specificity', epoch_metrics['specificity'], (epoch + 1))
                    writer.add_scalar('Metrics/Train/Sensitivity', epoch_metrics['sensitivity'], (epoch + 1))
                    writer.flush()
                elif phase == 'val':
                    writer.add_scalar('Loss/Validation/Cross Entropy Loss', epoch_loss, (epoch + 1))
                    writer.add_scalar('Metrics/Validation/Accuracy', epoch_acc, (epoch + 1))
                    writer.add_pr_curve('Metrics/Validation/PR-Curve', np.asarray(original_labels), np.asarray(predicted_labels), (epoch + 1))
                    writer.add_scalar('Metrics/Validation/F1-Score', epoch_metrics['f1_score'], (epoch + 1))
                    writer.add_scalar('Metrics/Validation/Precision', epoch_metrics['precision'], (epoch + 1))
                    writer.add_scalar('Metrics/Validation/Recall', epoch_metrics['recall'], (epoch + 1))
                    writer.add_scalar('Metrics/Validation/Specificity', epoch_metrics['specificity'], (epoch + 1))
                    writer.add_scalar('Metrics/Validation/Sensitivity', epoch_metrics['sensitivity'], (epoch + 1))
                    # logging.info(f'''Validation:
                    #     CE:    {epoch_loss}
                    # ''')
                    writer.flush()
                    val_epoch_loss.append(epoch_loss)
                    val_epoch_acc.append(epoch_acc)
                    if round(epoch_loss, 5) < prev_val_loss and round(epoch_acc, 5) > prev_val_acc and round(
                            epoch_metrics['f1_score'], 5) > prev_val_f1:
                        prev_val_loss = epoch_loss
                        prev_val_acc = epoch_acc
                        prev_val_f1 = epoch_metrics['f1_score']
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_res = f'''
                                                    Val Loss:   {epoch_loss}
                                                    Accuracy:   {epoch_acc}
                                                    Metrics:    {epoch_metrics}
                                                '''
                        writer.add_text('Best Results', best_res, (epoch + 1))
                        writer.flush()

        if save_cp:
            try:
                if not os.path.exists(dir_checkpoint):
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
            except OSError:
                pass
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(),
                           os.path.join(dir_checkpoint, f'modelp6{epoch + 1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved !')

        end_time = time()
        logging.info('Epoch took time: {}'.format(str(end_time - start_time)))

    writer.close()

    # Load n Save Best Model Weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(dir_checkpoint, 'modelp6_best_weights' + '.pth'))

    return train_epoch_loss, train_epoch_acc, train_epoch_metrics, val_epoch_loss, val_epoch_acc, val_epoch_metrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # empty cuda cache
    torch.cuda.empty_cache()
    logging.info(f'Using device {device}')

    model = networks.ProposedCNNModelP6()

    # We may Load state from checkpoints if something goes wrong while training

    model.to(device=device)

    # Arguments
    num_epochs = 20
    batch_size = 8
    learning_rate = 0.0001

    try:
        train_epoch_loss, train_epoch_acc, train_epoch_metrics, val_epoch_loss, val_epoch_acc, val_epoch_metrics = train_net(
            model=model,
            device=device,
            epochs=num_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            save_cp=True,
            optim='sgd'
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(dir_checkpoint, 'modelp6_1_INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
