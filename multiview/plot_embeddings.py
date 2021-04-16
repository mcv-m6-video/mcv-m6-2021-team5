import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images[:,0] = images[:,0].cuda()
            embeddings[k:k+len(images[:,0])] = model.get_embedding(images[:,0]).data.cpu().numpy()
            labels[k:k+len(images[:,0])] = target.numpy()
            k += len(images[:,0])
    return embeddings, labels


from datasets import aicityTriplet

def main():
    #triplet_train_dataset = aicityTriplet(train_dataset) # Returns triplets of images
    triplet_test_dataset = aicityTriplet(filenames_txt = "../datasets/test.txt", "../datasets/cars/S03")
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    #triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # Set up the network and training parameters
    from networks import EmbeddingNet, TripletNet
    from losses import TripletLoss

    margin = 1.
    embedding_net = EmbeddingNet()
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
    plot_embeddings(val_embeddings_tl, val_labels_tl)


if __name__ == "__main__":
    main()