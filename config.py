import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:{}".format(DEVICE))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

TRAINED_PATH = os.path.join(BASE_PATH, "trained")

EXPS_PATH = os.path.join(BASE_PATH, "out/experiments")

ATT_PATH = os.path.join(BASE_PATH, "out/attentions")

DATA_DIR = os.path.join(BASE_PATH, 'datasets')
