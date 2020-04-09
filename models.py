import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Models are subclasses of nn.Module and:
#   - Contain submodules (layers) as member variables defined in the `__init__` function
#   - Implement a 'forward' function that propagates the input through the network
class CustomModel(nn.Module):

    # '__init__' can take any number of arguments. Here, we pass the number of channels in the image (RGB=3)
    # as well as the number of classes in the output
    def __init__(self, in_channels, n_classes):
        super().__init__()

        # Member modules can be grouped together into larger units using nn.Sequential
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Member modules don't have to be grouped together at all. They can just be member variables
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)      # Take the maximum activity across all spatial dimensions
        self.fc1 = nn.Linear(in_features=128, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=n_classes)

        # Note: for a module's weights to be automatically included in the Model's
        # trainable parameters, it must be declared either as a direct member variable, or
        # it must be within a nn.Sequential/nn.ModuleList member variable.
        # For instance, if you write:
        #   self.layers = [nn.Conv2d(...), nn.Linear(...)]
        # PyTorch will not register these layers' weights as parameters of the Model,
        # even if you use the layers in the forward pass. So, they will not be updated during training!

    # 'forward' can take any number of inputs, and can return any number of outputs
    #
    # You can have ANY Python code and control flow you want within the forward function (if's, for's, etc.)
    # PyTorch creates computation graphs dynamically for each forward pass, so your model doesn't have to do the
    # same thing every time. This is particularly useful for things like RNN's
    def forward(self, image):
        x = self.features(image)            # Pass the input through all the convolutional layers
        x = self.maxpool(x)

        x = x.view(x.shape[0], -1)          # Flatten the height and width dimensions

        x = self.fc1(x)                     # Pass the convolutional features through the fully-connected layers
        x = self.fc2(x)

        # When training, the PyTorch cross-entropy loss adds a logsoftmax layer to your output,
        # so you should just return the raw logits without applying a softmax
        if not self.training:
            x = F.softmax(x, dim=1)     # The nn.functional library provides many parameter-less layers

        return x


# You can build your model off of the existing pretrained models from the torchvision package.
# This can be useful when you don't want to train at all, or when you want to fine-tune a pretrained model.
class PretrainedModel(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        alexnet = models.alexnet(pretrained=True)
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=n_classes)
        )

    def forward(self, image):
        x = self.features(image)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return x

    # If you don't want to fine-tune the whole model, you simply
    # don't have to return all of the model's parameters to the optimizer.
    # In this case, I'm overriding the base class's existing `parameters()` function for simplicity.
    # If you DO want to fine-tune the whole model, simply delete this function
    def parameters(self):
        return self.features.parameters()


class AlexNetDesiredLayer(nn.Module):

    def __init__(self, feature_name):
        super().__init__()
        assert feature_name in ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5',
                                'pool', 'fc_1', 'fc_2', 'fc_3']

        self.feature_name = feature_name
        base = models.alexnet(pretrained=True)

        self.conv_1 = base.features[:3]
        self.conv_2 = base.features[3:6]
        self.conv_3 = base.features[6:8]
        self.conv_4 = base.features[8:10]
        self.conv_5 = base.features[10:]
        self.avgpool = base.avgpool
        self.fc_1 = base.classifier[:3]
        self.fc_2 = base.classifier[3:6]
        self.fc_3 = base.classifier[6:]

    def forward(self, stimuli):
        x = self.conv_1(stimuli)
        if 'conv_1' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_2(x)
        if 'conv_2' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_3(x)
        if 'conv_3' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_4(x)
        if 'conv_4' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_5(x)
        if 'conv_5' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        if 'pool' == self.feature_name: return x
        x = self.fc_1(x)
        if 'fc_1' == self.feature_name: return x
        x = self.fc_2(x)
        if 'fc_2' == self.feature_name: return x
        x = self.fc_3(x)
        if 'fc_3' == self.feature_name: return x
        return None


class AlexNetConv3Channel2(nn.Module):

    def __init__(self):
        super().__init__()

        base = models.alexnet(pretrained=True)

        self.conv_1 = base.features[:3]
        self.conv_2 = base.features[3:6]
        self.conv_3 = base.features[6:8]

    def forward(self, stimuli):
        x = self.conv_1(stimuli)
        x = self.conv_2(x)
        x = self.conv_3(x)
        conv3_channel2 = x[:, 1]    # All batches, 2nd channel (index=1), all spatial dimensions
        # conv3_channel2_unitheight4width2 = x[:, 1, 3, 1]  # Here's how you would get just a single unit in the layer
        return conv3_channel2
