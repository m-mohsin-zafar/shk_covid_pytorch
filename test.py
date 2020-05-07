import logging
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import utilities as utils

from torch.utils.data import DataLoader

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
#                                       Testing the Model
# ------------------------------------------------------------------------------------
def test_net(model, device, cp_path, test_df):

    # We load the model and put it in evaluation mode
    model.load_state_dict(torch.load(cp_path))
    model.eval()

    test_set = CovidDataset(
        root=dir_data_root,
        inp_df=test_df,
        transformations=utils.get_transformations(for_train=False)
    )
    test_set_dl = DataLoader(
        test_set,
        batch_size=4,
        shuffle=True
    )

    set_len = len(test_set)
    desc = 'Testing Phase'

    total = 0
    correct_preds = 0
    original_labels = []
    predicted_labels = []
    with tqdm(total=set_len, desc=desc, unit='img') as pbar:
        for batch in test_set_dl:
            _, images, labels = batch  # Let's get images and labels for each batch
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                total += images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_preds += (predicted == labels).sum().item()
                original_labels += labels.data.cpu().numpy().tolist()
                predicted_labels += predicted.data.cpu().numpy().tolist()

            pbar.update(images.shape[0])

    test_acc = (correct_preds / total) * 100
    test_metrics = utils.evaluate_metrics(original_labels, predicted_labels)
    return test_acc, test_metrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # empty cuda cache
    torch.cuda.empty_cache()
    logging.info(f'Using device {device}')

    model = networks.CovidRENet()

    # First we read 'labels.csv' and construct input dataframe
    inp_df = pd.read_csv(correct_labels)

    # Now we get the stratified DataFrames for input to our Dataset Objects
    _, _, test_df = utils.get_stratified_train_val_test_sets(inp_df=inp_df, seed=seed_val)

    logging.info(f'--------------This is Testing Phase--------------')

    model.to(device=device)

    cp_path = os.path.join(dir_checkpoint, 'CovidRENet__best_weights.pth')

    test_acc, test_metrics = test_net(model, device, cp_path, test_df=test_df)

    logging.info(f'''   Test Accuracy      :   {test_acc}
                        Test Metrics       :   {test_metrics}
                    ''')
