import torch
from torch import nn
import torch.nn.functional as F
import time
import copy
import torchvision.models as models
from utility import process_image

##class for classifier
class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers

        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


def validation(model, validloader, criterion):
    accuracy = 0
    valid_loss = 0
    for images, labels in validloader:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        valid_loss += criterion(output, labels).item()

        ##calculat the accuracy
        ps = torch.exp(output)

        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return valid_loss, accuracy



def train(model, trainloader, validloader, criterion, optimizer, epochs, print_every = 20):
    since = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    #define some tracking parameters
    running_loss = 0
    steps = 0

    model.to(device)

    for e in range(epochs):

        start = time.time()

        for images, labels in trainloader:

                steps+=1

                optimizer.zero_grad()

                images, labels = images.to(device), labels.to(device)

                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()

                    with torch.no_grad():

                        valid_loss, accuracy = validation(model, validloader, criterion)


                    print("Epoch: {}/{}".format(e+1, epochs),
                          "Training Loss: {:.3f}".format(running_loss/print_every),
                          "Valid Loss: {:.3f}".format(valid_loss/len(validloader)),
                          "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))

                    if accuracy > best_acc:
                        best_acc = accuracy
                        best_model_wts = copy.deepcopy(model.state_dict())

                    running_loss = 0
                    model.train()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc/len(validloader)))


    model.load_state_dict(best_model_wts)
    return model


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    #download the pretrain arch
    pretrain_model = getattr(models, checkpoint['model_arch'])
    model = pretrain_model(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    #rebuild the classifier
    model.classifier = Classifier(25088, 102, [checkpoint['hidden_units']], drop_p=0.2)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def predict(image_path, model, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    #process images
    image = process_image(image_path)
    log_ps = model(image.view(1,3,224,224))
    ps = torch.exp(log_ps)
    probs, idx = ps.topk(topk, dim=1)

    probs = probs.detach().numpy().flatten()
    idx = idx.detach().numpy().flatten()

    classes = [dict(map(reversed, model.class_to_idx.items()))[i] for i in idx]
    class_name = [cat_to_name[str(x)] for x in classes]

    return probs, class_name
