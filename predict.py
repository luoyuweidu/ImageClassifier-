from utility import load_process_data
import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn
from torch import optim
import time
from collections import OrderedDict
import copy
from workspace_utils import active_session
from model_func import Classifier, load_checkpoint, predict
import argparse
import sys
import json


#set up a parser
parser = argparse.ArgumentParser(
        description='parse for prediction app'
)

parser.add_argument('--top_k',
                    action='store',
                    dest='top_k',
                    help='show top k prediction')

parser.add_argument('--category_names',
                    action='store',
                    dest='category_names',
                    help='mapping to category')

parser.add_argument('--gpu',
                    action='store_true',
                    default=True,
                    help='mapping to category')


#parse system input
args = parser.parse_args(sys.argv[3:])
top_k = args.top_k if args.top_k else 3
category_names = args.category_names if args.category_names else 'cat_to_name.json'
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
checkpoint_path = sys.argv[2]
image_input = sys.argv[1]

#load the checkpoint
model = load_checkpoint(checkpoint_path)


#load image and process
probs, class_names = predict(image_input, model, cat_to_name, top_k)

#print prediction result
print('Top {0} classes and probabilities: {1}'.format(top_k, list((zip(class_names,probs)))))
