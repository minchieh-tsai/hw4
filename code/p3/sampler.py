import torch
import numpy as np
from dataset_1 import MiniDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision


class CategoriesSampler():

	def __init__(self, n_batch, total_cls , n_cls , n_per):
		self.n_batch = n_batch
		self.total_cls = total_cls
		self.n_cls = n_cls
		self.n_per = n_per
		self.classes = 0

		# c是class,
		self.iters = []
		batch_size = n_cls  # N way
		while len(self.iters) < n_batch:
			self.classes = np.arange(total_cls)
			np.random.shuffle(self.classes)
			for i in range (total_cls // batch_size):
				self.iters.append(self.classes[i * batch_size: (i + 1) * batch_size] )
				if len(self.iters) == n_batch: break    # 如果再 for 裡面就達到 batch size，提早跳出


	def __len__(self):
		return self.n_batch

	def __iter__(self):

		for self.classes in self.iters:
			batch =[]
			for one_class in self.classes:
				img_idx = np.random.randint(0, 600, self.n_per)
				img_idx = one_class * 600 + img_idx
				
				batch.append(torch.tensor(img_idx, dtype=torch.int))

			batch = torch.stack(batch).t().reshape(-1)
			yield batch
				



# 		for i_batch in range(self.n_batch):
# 			batch = []
# 			classes = torch.randperm(len(self.m_ind))[:self.n_cls]
# 			for c in classes:
# 				l = self.m_ind[c]
# 				pos = torch.randperm(len(l))[:self.n_per]
# 				batch.append(l[pos])
# 			batch = torch.stack(batch).t().reshape(-1)
# 			yield batch


def imshow(image):
    npimg = image.numpy()
    nptran = np.transpose(npimg, (1, 2, 0))
    plt.figure()
    plt.imshow(nptran)


if __name__ == '__main__':
	n_batch = 500
	shot = 5
	query = 1
	train_way = 5
	total_cls = 64

	trainset = MiniDataset(csv_path='..\\..\\hw4_data\\train.csv', data_dir='..\\..\\hw4_data\\train')
	train_sampler = CategoriesSampler( n_batch, total_cls , train_way, shot + query)
	train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler)

# 	dataIter_train = train_loader.__iter__()
# 	var = dataIter_train.next()
# 	imshow(torchvision.utils.make_grid(var[0]))
# 	print(var[1])
	
	for i , (img,label) in enumerate(train_loader):
		print(i,label)


	# for i in train_sampler:
	# 	print(i)
	# 	break
