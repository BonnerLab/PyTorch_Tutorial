import torch
from torch.utils.data import DataLoader
from torchvision import models
from CustomDataset import CustomDataset


def run_dataset(model, dataloader):
    with torch.no_grad():   # Do not make dynamic graph, since not backpropagating. Uses much less RAM
        for batch_num, images in enumerate(dataloader):   # For every batch in the dataset
            print('Running batch %d/%d' % (batch_num + 1, len(dataloader)))

            if torch.cuda.is_available():   # Put inputs on GPU, if one is available
                images = images.cuda()

            # Run the model on these images
            # For every image in the batch, predictions contain the probabilities of each class
            # predictions shape is BATCH_SIZExNUM_CLASSES (NUM_CLASSES is 1000 for AlexNet pre-trained on ImageNet)
            predictions = model(images)


if __name__ == '__main__':
    alexnet = models.alexnet(pretrained=True)   # Grab AlexNet pre-trained on ImageNet from standard Pytorch models
    alexnet.eval()  # Certain layers (e.g. BatchNorm and Dropout) behave differently in training vs. evaluation mode
    if torch.cuda.is_available():   # Put model on GPU, if one is available
        alexnet.cuda()

    test_set = CustomDataset(data_dir='data', resolution=(224, 224))
    test_loader = DataLoader(dataset=test_set, batch_size=1)

    print('Running dataset')
    run_dataset(alexnet, test_loader)
    print('Finished')
