# ModiFace PyTorch Tutorial
This respository provides:

- A high-level overview of how PyTorch works, 
including the typical structure of projects and brief remarks
about how it works (and differs from static graph computation libraries
such as TensorFlow).
- A working example of training a model on MNIST data, including
working with PyTorch Modules, Datasets, fine-tunning models, and
augmenting image data.

If you're already convinced of the benefits of PyTorch and
know roughly how it works, feel free to skip the rest of this
README and jump straight into the code.

### Requirements

- PyTorch 0.4 (see PyTorch website for installation instructions)
- TensorboardX (can install through pip or conda using this name)

### How PyTorch works (very briefly)

##### How most ML libraries work

Most machine learning libraries work by allowing you to 
define static computation graphs, which then run on an 
efficient backend (usually C++). For example, you can
run a model in tensorflow by writing something like:

```predictions = tf.Session().run([model_output_node])```

Although efficient, defining static computation graphs 
provides two major disadvantages:

- It becomes very difficult to write a model with control 
and general higher level code (e.g. your model does two
separate things depending on its input, or executes custom
logic in something like a for-loop). On top of requiring you
code in a fundamentally different way, this results in other
practical inconveniences. For instance, TensorFlow often
necessitates writing completely different graphs for training
and inference. Another task where the difficulties of static
graph computation become aparent are when coding RNN's. The 
intuitive way to code RNN's is to have inputs propagate through 
a for-loop. Since this is impossible (or at least, very
convoluted) to achieve in TensorFlow, the API around RNN's in 
TensorFlow must provide many functions to do things such as 
unroll a sequence of inputs, and inference type with arbirarily 
long sequence lengths is even more of a pain to work with.
- Your model and whatever other nodes are part of the computation
graph execute completely outside of your coding environment.
When you have run-time errors in your computation graph, you
don't get line numbers of when something went wrong. You get
error messages from the backend which are often hard to reconcile
with the code that you wrote. Moreover, when there isn't any error
but training still doesn't seem to be working, you can't easily
just use a debugger to walk through data propagating through your
model. To be fair, TensorFlow does provide a command-line
tool that lets you walk through your model, but it is difficult
to use when compared to simply working in an IDE with your
favourite debugger.

##### How PyTorch differs

In contrast to libraries that use static computation graphs,
PyTorch builds a dynamic graph every time your model processes
inputs. This dynamic graph includes all the necessary information
to perform backpropagation, such as :

- The sequence of operations that were performed
- The intermediate values (activations) of your layeres

At face value, this may sound very inefficient in terms
of computation time and memory performance, but due to some
magic by the developers of PyTorch training ends up being
comporable to (and by some accounts, faster) than static
computation graph libraries.

It is important to note that this doesn't mean every bit
of code is executing locally in Python. In fact, the
important layers and operations execute on a C++ backend as
well. The difference is that the entire computation graph
isn't executing on that backend, and therefore permits much
more integration with Python.

### Project structure in PyTorch

Perhaps more important than the advantages described above, 
the PyTorch API was designed very intuitively, drastically
reducing boilerplate code and making project structure
intuitive. This difference cannot be overstated. I think that
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
