# Bonner Lab PyTorch Tutorial
This respository provides:

- A high-level overview of how PyTorch works, 
including the typical structure of projects and brief remarks
about how it works.
- A working example of training a model on MNIST data, including
working with PyTorch Modules, Datasets, fine-tunning models, and
augmenting image data.
- Jupyter Notebooks showing how to use models for basic purposes
(e.g. extract activations from a desired network layer/channel).

### Requirements

- PyTorch+Torchvision 
`pip install torch torchvision`
- TensorboardX (for visualizing training curves) 
`pip install tensorboardx`

### Project structure in PyTorch

The PyTorch API was designed very intuitive, minimizing boilerplate code. 
This difference from alternative frameworks cannot be overstated. I think that
you'll find understanding a good PyTorch project is a _much_
easier task than understanding a functionally equivalent
TensorFlow (or even Keras) project.

##### The model

You define your model in PyTorch as a class that inherits
from the `torch.nn.Module` superclass. In fact, every layer
in PyTorch is a subclass of `torch.nn.Module`, so you can
think of your model as one big layer. The only requirements
for the model to work correctly are:

- It must have it's submodules (layers) as direct member variables.
This is necessary for PyTorch to recognize the layers' weights as
parameters of the model.
- It must implement a `forward(self, *args)` function that takes inputs and
returns on or more outputs. This is where your model's
connections are essentially being specified. Any arbiraty
Python code and control flow can be used in this function.

##### The dataset

This is the class that handles the loading of one sample
of data at a time. It inherits from `torch.utils.data.Dataset`
and must implement two functions:

- `__len__(self)` simply returns the number of samples
contained in your dataset
- `__getitem__(self, item)` should return the input/label
tensors at index `item`. This is where you load images
and do any necessary data augmentation.

The dataset class that you define works in conjunction
with the `torch.utils.data.DataLoader` class, which takes
as it's argument a dataset object, as well as parameters
about the batchsize, whether or not to shuffle the data
after each epoch, how many threads to use for loading the
data, etc.

##### The training loop

Typically, your training script instantiates the model,
instantiates training and test/validation data loaders,
instantiates an optimizer (e.g. Adam) with the model's parameters
and then begins the training loop. At its most basic level,
(without evaluation/logging) the training loop looks like this:

```
for epoch in n_epochs:
    for input, label in trainloader:
        predictions = model(input)
        loss = compute_loss(predictions, labels)
        optimizer.zero_grad()   # clear parameter gradients from previous batch
        loss.backward()         # backprop to compute parameter gradients
        optimizer.step()        # update weights using the parameter gradients
```

### The rest of this repository

The rest of this repository forms a complete working
example training on the MNIST dataset. It covers:

- Building a model from scratch
- Building a model off of PyTorch's pretrained models
- Writing a dataset that does data augmentation during training
- Training/evaluation
- Saving/loading trained models

Hopefully, the comments should be sufficient to understand
the project as a whole. The two scripts that you can run
are `train.py` and `evaluate.py` (neither of which takes
any arguments). The recommended entry-point to walk through
all of the code is to start at `train.py`.

Jupyter Notebooks are also provided to show minimally-functining examples
of tasks useful for research purposes (e.g. extracting the activations at a specific layer).
