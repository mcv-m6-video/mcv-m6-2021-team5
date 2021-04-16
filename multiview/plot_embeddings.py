import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from networks import Net

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt

# mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#               '#bcbd22', '#17becf']

def plot_embeddings(num_classes, embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(num_classes):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    num_classes_list = list(range(num_classes))
    plt.legend(num_classes_list)
    plt.show()

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        # embedding layer --> middle layer with size 512
        embeddings = np.zeros((len(dataloader.dataset), 512))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        print("Length dataset: " + str(len(dataloader.dataset)))
        # One target (from anchor) per every triplet of images
        for images, target in dataloader:
            print(np.shape(images))
            print(np.shape(target))
            if cuda:
                images[0] = images[0].cuda()
            # print(images[0])
            # print(len(images[0]))
            print("Processing image...: " + str(k))
            embeddings[k:k+len(images[0])] = model.get_embedding(images[0]).data.cpu().numpy()
            labels[k:k+len(images[0])] = np.array(target)
            # Adds one to k, since we only read one image at a time (batch_size = 1)
            k += len(images[0])
    return embeddings, labels


from datasets import aicityTriplet

def main():
    # Transformation for dataset
    transform = transforms.Compose([
						   transforms.Resize((96,96)),
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
					        0.229, 0.224, 0.225])
					    ])
    #triplet_train_dataset = aicityTriplet(train_dataset) # Returns triplets of images
    triplet_test_dataset = aicityTriplet(filenames_txt = "../datasets/test.txt", image_dir = "../datasets/cars/S03", transform = transform)
    num_classes = len(set(triplet_test_dataset.labels_set))
    print("Num classes: " + str(num_classes))
    batch_size = 1
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    #triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # Set up the network and training parameters
    from networks import EmbeddingNet, TripletNet
    from losses import TripletLoss

    margin = 1.
    embedding_net = Net()
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 100


    #fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    # train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
    # plot_embeddings(train_embeddings_tl, train_labels_tl)
    val_embeddings_tl, val_labels_tl = extract_embeddings(triplet_test_loader, model)
    # Embeddings have a size of (1, 512) --> Apply PCA or any dimensionality reduction technique!
    plot_embeddings(num_classes, val_embeddings_tl, val_labels_tl)


if __name__ == "__main__":
    main()