# MiVOLO Classifier 

import argparse
import logging
import os
import sys
import ultralytics
import cv2
import torch
import yt_dlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics.utils.plotting import colors

# Scripts
from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging
from cf_utils import prepare_img_for_mivolo
from cf_utils import normalize_0_1

_logger = logging.getLogger("inference")

# Helper function
def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--input", type=str, default=None, required=True, help="image file or folder with images")
    parser.add_argument("--output", type=str, default=None, required=True, help="folder for output results")
    parser.add_argument("--detector-weights", type=str, default=None, required=True, help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")

    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"
    )

    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")

    return parser

detector_weights = "./mivolo_models/yolov8x_person_face.pt"
checkpoint =  "./mivolo_models/model_imdb_cross_person_4.22_99.46.pth.tar"
device = "cpu"
input = "./celeba_hq_dataset/CelebA-HQ-img" 

parser = get_parser()
args = parser.parse_args(
    [
        "--input",f"""{input}""", 
        "--detector-weights", f"""{detector_weights}""", 
        "--checkpoint", f"""{checkpoint}""", 
        "--device", f"""{device}""",
        "--output", 'output',
        "--with-persons",
        "--draw"
        
    ]
)

predictor = Predictor(args, verbose=False)

def mivolo_inference(torch_img, return_age = False):
  torch_img = normalize_0_1(torch_img)
  cv2_img = prepare_img_for_mivolo(torch_img)
  detected_objects, out_im = predictor.recognize(cv2_img)
  if detected_objects.n_faces != 0:
    gender_string = detected_objects.genders[0]
    mivolo_gender_score = detected_objects.gender_scores[0]
    age = detected_objects.ages[0]
    if gender_string == "male":
      mivolo_gender_prob = 1 - mivolo_gender_score
    elif gender_string == "female":
      mivolo_gender_prob = mivolo_gender_score
  else:
    mivolo_gender_prob = np.inf
    age = np.inf
  if return_age:
    return mivolo_gender_prob, age
  else:
    return mivolo_gender_prob