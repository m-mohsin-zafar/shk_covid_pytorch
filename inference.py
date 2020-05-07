import logging
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import utilities as utils

from torch.utils.data import DataLoader

from datasets import InferenceDataset
import networks

torch.cuda.current_device()

# ------------------------------------------------------------------------------------
#                                       Some Paths to be used
# ------------------------------------------------------------------------------------
dir_cwd = os.getcwd()
dir_inference_data_root = os.path.join(dir_cwd, 'data', 'inference_data')
dir_checkpoint = os.path.join(dir_cwd, 'checkpoints')


def get_model(cp_name='CovidRENet__best_weights', model_name='covidrenet'):
    chk_pt = os.path.join(dir_checkpoint, cp_name)
    if model_name == 'covidrenet':
        model = networks.CovidRENet()
    else:
        model = networks.CustomVGG16()
    model.load_state_dict(torch.load(chk_pt, map_location='cpu'), strict=False)
    model.eval()
    return model


def get_class_to_name_mapping(cls):
    class_to_name = {
        0: 'Non Corona',
        1: 'Corona'
    }
    return class_to_name[cls]


model = get_model()


def get_predictions():

    inference_set = InferenceDataset(
        root=dir_inference_data_root,
        fnames=os.listdir(dir_inference_data_root),
        transformations=utils.get_transformations(for_train=False)
    )
    inference_set_dl = DataLoader(
        inference_set,
        batch_size=1,
        shuffle=False
    )
    outputs = []
    try:
        for batch in inference_set_dl:
            _, images = batch
            images = images.cpu()
            output = model.forward(images)
            conf, prediction = output.max(1)
            xray_class = prediction.item()
            xray_class_name = get_class_to_name_mapping(cls=xray_class)

            outputs.append((xray_class, xray_class_name))

    except Exception:
        return 0, 'error!'

    return outputs
