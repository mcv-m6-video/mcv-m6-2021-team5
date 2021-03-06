import numpy as np
import xml.etree.ElementTree as ET
import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


def make_data(path):
	tree = ET.parse(path,
					parser=ET.XMLParser(encoding='iso-8859-5'))
	root = tree.getroot()

	labels_list = []
	data = []
	for item in root.findall("Items"):
		for item1 in item.findall("Item"):
			attrib = item1.attrib
			data.append(attrib)
			labels_list.append(int(attrib['vehicleID']))
	
	labels_array = np.array(labels_list)
	return data, labels_array


class SiameseMNIST(Dataset):
	"""
	Train: For each sample creates randomly a positive or a negative pair
	Test: Creates fixed pairs for testing
	"""

	def __init__(self, mnist_dataset):
		self.mnist_dataset = mnist_dataset

		self.train = self.mnist_dataset.train
		self.transform = self.mnist_dataset.transform

		if self.train:
			self.train_labels = self.mnist_dataset.train_labels
			self.train_data = self.mnist_dataset.train_data
			self.labels_set = set(self.train_labels.numpy())
			self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
									 for label in self.labels_set}
		else:
			# generate fixed pairs for testing
			self.test_labels = self.mnist_dataset.test_labels
			self.test_data = self.mnist_dataset.test_data
			self.labels_set = set(self.test_labels.numpy())
			self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
									 for label in self.labels_set}

			random_state = np.random.RandomState(29)

			positive_pairs = [[i,
							   random_state.choice(
								   self.label_to_indices[self.test_labels[i].item()]),
							   1]
							  for i in range(0, len(self.test_data), 2)]

			negative_pairs = [[i,
							   random_state.choice(self.label_to_indices[
								   np.random.choice(
									   list(self.labels_set -
											set([self.test_labels[i].item()]))
								   )
							   ]),
							   0]
							  for i in range(1, len(self.test_data), 2)]
			self.test_pairs = positive_pairs + negative_pairs

	def __getitem__(self, index):
		if self.train:
			target = np.random.randint(0, 2)
			img1, label1 = self.train_data[index], self.train_labels[index].item(
			)
			if target == 1:
				siamese_index = index
				while siamese_index == index:
					siamese_index = np.random.choice(
						self.label_to_indices[label1])
			else:
				siamese_label = np.random.choice(
					list(self.labels_set - set([label1])))
				siamese_index = np.random.choice(
					self.label_to_indices[siamese_label])
			img2 = self.train_data[siamese_index]
		else:
			img1 = self.test_data[self.test_pairs[index][0]]
			img2 = self.test_data[self.test_pairs[index][1]]
			target = self.test_pairs[index][2]

		img1 = Image.fromarray(img1.numpy(), mode='L')
		img2 = Image.fromarray(img2.numpy(), mode='L')
		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
		return (img1, img2), target

	def __len__(self):
		return len(self.mnist_dataset)


class TripletMNIST(Dataset):
	"""
	Train: For each sample (anchor) randomly chooses a positive and negative samples
	Test: Creates fixed triplets for testing
	"""

	def __init__(self, mnist_dataset):
		self.mnist_dataset = mnist_dataset
		self.train = self.mnist_dataset.train
		self.transform = self.mnist_dataset.transform

		if self.train:
			self.train_labels = self.mnist_dataset.train_labels
			self.train_data = self.mnist_dataset.train_data
			self.labels_set = set(self.train_labels.numpy())
			print(self.train_labels)
			print(len(self.train_labels))
			print(self.labels_set)
			self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
									 for label in self.labels_set}
			print(self.label_to_indices)

		else:
			self.test_labels = self.mnist_dataset.test_labels
			self.test_data = self.mnist_dataset.test_data
			# generate fixed triplets for testing
			self.labels_set = set(self.test_labels.numpy())
			self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
									 for label in self.labels_set}

			random_state = np.random.RandomState(29)

			triplets = [[i,
						 random_state.choice(
							 self.label_to_indices[self.test_labels[i].item()]),
						 random_state.choice(self.label_to_indices[
							 np.random.choice(
								 list(self.labels_set -
									  set([self.test_labels[i].item()]))
							 )
						 ])
						 ]
						for i in range(len(self.test_data))]
			self.test_triplets = triplets

	def __getitem__(self, index):
		if self.train:
			img1, label1 = self.train_data[index], self.train_labels[index].item(
			)
			positive_index = index
			while positive_index == index:
				positive_index = np.random.choice(
					self.label_to_indices[label1])
			negative_label = np.random.choice(
				list(self.labels_set - set([label1])))
			negative_index = np.random.choice(
				self.label_to_indices[negative_label])
			img2 = self.train_data[positive_index]
			img3 = self.train_data[negative_index]
		else:
			img1 = self.test_data[self.test_triplets[index][0]]
			img2 = self.test_data[self.test_triplets[index][1]]
			img3 = self.test_data[self.test_triplets[index][2]]

		img1 = Image.fromarray(img1.numpy(), mode='L')
		img2 = Image.fromarray(img2.numpy(), mode='L')
		img3 = Image.fromarray(img3.numpy(), mode='L')
		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
			img3 = self.transform(img3)
		return (img1, img2, img3), []

	def __len__(self):
		return len(self.mnist_dataset)


class TripletVeriDataset(Dataset):
	"""
	Train: For each sample (anchor) randomly chooses a positive and negative samples
	Test: Creates fixed triplets for testing
	"""

	def __init__(self, root_dir, xml_path, transform):

		self.train_data, self.train_labels  = make_data(xml_path)
		self.transform = transform
		self.root_dir = root_dir
		self.labels_set = set(self.train_labels)

		self.label_to_indices = {label: np.where(self.train_labels == label)[0]
									for label in self.labels_set}
	
	def __getitem__(self, index):
		img_name1, label1 = self.train_data[index]['imageName'], int(self.train_data[index]['vehicleID'])
		positive_index = index
		while positive_index == index:
			positive_index = np.random.choice(
				self.label_to_indices[label1])
		negative_label = np.random.choice(
			list(self.labels_set - set([label1])))
		negative_index = np.random.choice(
			self.label_to_indices[negative_label])
		img_name2 = self.train_data[positive_index]['imageName']
		img_name3 = self.train_data[negative_index]['imageName']
	
		img1 = Image.open(os.path.join(self.root_dir, img_name1))
		img2 = Image.open(os.path.join(self.root_dir, img_name2))
		img3 = Image.open(os.path.join(self.root_dir, img_name3))
		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
			img3 = self.transform(img3)
		return (img1, img2, img3), []

	def __len__(self):
		return len(self.train_data)


class BalancedBatchSampler(BatchSampler):
	"""
	BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
	Returns batches of size n_classes * n_samples
	"""

	def __init__(self, labels, n_classes, n_samples):
		self.labels = labels
		self.labels_set = list(set(self.labels.numpy()))
		self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
								 for label in self.labels_set}
		for l in self.labels_set:
			np.random.shuffle(self.label_to_indices[l])
		self.used_label_indices_count = {label: 0 for label in self.labels_set}
		self.count = 0
		self.n_classes = n_classes
		self.n_samples = n_samples
		self.n_dataset = len(self.labels)
		self.batch_size = self.n_samples * self.n_classes

	def __iter__(self):
		self.count = 0
		while self.count + self.batch_size < self.n_dataset:
			classes = np.random.choice(
				self.labels_set, self.n_classes, replace=False)
			indices = []
			for class_ in classes:
				indices.extend(self.label_to_indices[class_][
							   self.used_label_indices_count[class_]:self.used_label_indices_count[
								   class_] + self.n_samples])
				self.used_label_indices_count[class_] += self.n_samples
				if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
					np.random.shuffle(self.label_to_indices[class_])
					self.used_label_indices_count[class_] = 0
			yield indices
			self.count += self.n_classes * self.n_samples

	def __len__(self):
		return self.n_dataset // self.batch_size


class IndianFaces(Dataset):
	"""
	Train: For each sample (anchor) randomly chooses a positive and negative samples
	Test: Creates fixed triplets for testing
	"""

	def __init__(self, mnist_dataset):
		self.mnist_dataset = mnist_dataset
		self.train = self.mnist_dataset.train
		self.transform = self.mnist_dataset.transform

		if self.train:
			self.train_labels = self.mnist_dataset.train_labels
			self.train_data = self.mnist_dataset.train_data
			self.labels_set = set(self.train_labels.numpy())
			self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
									 for label in self.labels_set}

		else:
			self.test_labels = self.mnist_dataset.test_labels
			self.test_data = self.mnist_dataset.test_data
			# generate fixed triplets for testing
			self.labels_set = set(self.test_labels.numpy())
			self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
									 for label in self.labels_set}

			random_state = np.random.RandomState(29)

			triplets = [[i,
						 random_state.choice(
							 self.label_to_indices[self.test_labels[i].item()]),
						 random_state.choice(self.label_to_indices[
							 np.random.choice(
								 list(self.labels_set -
									  set([self.test_labels[i].item()]))
							 )
						 ])
						 ]
						for i in range(len(self.test_data))]
			self.test_triplets = triplets

	def __getitem__(self, index):
		if self.train:
			img1, label1 = self.train_data[index], self.train_labels[index].item(
			)
			positive_index = index
			while positive_index == index:
				positive_index = np.random.choice(
					self.label_to_indices[label1])
			negative_label = np.random.choice(
				list(self.labels_set - set([label1])))
			negative_index = np.random.choice(
				self.label_to_indices[negative_label])
			img2 = self.train_data[positive_index]
			img3 = self.train_data[negative_index]
		else:
			img1 = self.test_data[self.test_triplets[index][0]]
			img2 = self.test_data[self.test_triplets[index][1]]
			img3 = self.test_data[self.test_triplets[index][2]]

		img1 = Image.fromarray(img1.numpy(), mode='L')
		img2 = Image.fromarray(img2.numpy(), mode='L')
		img3 = Image.fromarray(img3.numpy(), mode='L')
		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
			img3 = self.transform(img3)
		return (img1, img2, img3), []

	def __len__(self):
		return len(self.mnist_dataset)

class aicityTriplet(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, filenames_txt, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(filenames_txt) as f:
        	self.filename_list = [line.rstrip() for line in f]

        self.labels = []
        for filename in self.filename_list:
            self.labels.append(int(filename.split('_')[0]))
        
        self.labels_set = set(self.labels)
        # print(self.labels_set)
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                            for label in self.labels_set}
        # print(self.label_to_indices)

    def __getitem__(self, index):

        img1_path = os.path.join(self.image_dir, self.filename_list[index])
        img1 = Image.open(img1_path)
		# Anchor image label
        label1 = self.labels[index]

        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])

        negative_label = np.random.choice(list(self.labels_set - set([label1])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])

        img2_path = os.path.join(self.image_dir,self.filename_list[positive_index])
        img2 = Image.open(img2_path)
        img3_path = os.path.join(self.image_dir,self.filename_list[negative_index])
        img3 = Image.open(img3_path)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
		# Return triplet of images and label of anchor image for the sake of plotting
        #return (np.array(img1), np.array(img2), np.array(img3)), label1
        return (np.array(img1), np.array(img2), np.array(img3)), []

    def __len__(self):
        return len(self.filename_list)