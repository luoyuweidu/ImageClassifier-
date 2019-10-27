####import packages
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
from model_func import Classifier, train, validation
import argparse
import sys
import os


###define a parse
parser = argparse.ArgumentParser(
        description='parse for application'
)

parser.add_argument('--save_dir',
                    action='store',
                    dest='save_directory',
                    help='directory to save the model')

parser.add_argument('--arch',
                    action='store',
                    dest='arch',
                    help='model architecture')

parser.add_argument('--learning_rate',
                    action='store',
                    dest='learning_rate',
                    type=float,
                    help='learning rate')

parser.add_argument('--hidden_units',
                    action='store',
                    dest='hidden_units',
                    type=int,
                    help='number of hidden units')

parser.add_argument('--epochs',
                    action='store',
                    dest='epochs',
                    type=int,
                    help='number of epochs')

parser.add_argument('--gpu',
                    action='store_true',
                    default=False)

#parse system argument
args = parser.parse_args(sys.argv[2:])

save_directory = args.save_directory if args.save_directory else os.getcwd()
arch = args.arch if args.arch else "vgg13"
learning_rate = args.learning_rate if args.learning_rate else 0.003
hidden_units = args.hidden_units if args.hidden_units else 500
epochs = args.epochs if args.epochs else 5



###read data
data_dir = sys.argv[1]
images_datasets, dataloaders = load_process_data(data_dir)
print("Loaded data from {0}".format(data_dir))


###download pretrained model and build classifier
pretrain_model = getattr(models, arch)
model = pretrain_model(pretrained=True)


for param in model.parameters():
    param.requires_grad = False

model.classifier = Classifier(25088, 102, [hidden_units], drop_p=0.2)
print(model)
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu  else "cpu")

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

model = train(model, dataloaders['train'], dataloaders['valid'], criterion, optimizer, epochs = epochs, print_every = 20)



###save the model
model.class_to_idx = images_datasets['train'].class_to_idx

checkpoint = {'class_to_idx': model.class_to_idx,
              'state_dict':model.state_dict(),
              'optimizer_state_dict':optimizer.state_dict(),
              'epochs': 5,
              'hidden_units':hidden_units,
              'model_arch':arch}

print('Saving the model')
torch.save(checkpoint, os.path.join(save_directory,'checkpoint.pth'))
