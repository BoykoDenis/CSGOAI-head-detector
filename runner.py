import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import numpy as np
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
load_m = True
epochs = 10
lr = 0.01
dataset_size = 1746
batch_size = 1
n_features = 100
input_chanels = 1
dataset_path = "D:\\Datasets\\Head_hunt\\"

transform_train = transforms.Compose([transforms.ToTensor(),
									  transforms.Normalize((0.5,),
									  					   (0.5,))
									])



rn = random.randrange(1, 1000)
img = Image.open(dataset_path + "head (" + str(97) + ").png" ).convert("L")
img = transform_train(img)
img = img.unsqueeze(0)

plt.imshow(img[0][0])
plt.show()

class Head_hunter(nn.Module):
	def __init__(self, n_features, input_chanels, batch_size):
		super().__init__()
		self.conv1 = nn.Conv2d(input_chanels, n_features, kernel_size = 5, stride = 1, padding = 2)
		self.conv2 = nn.Conv2d(n_features, n_features, kernel_size = 5, stride = 1, padding = 2)
		self.conv3 = nn.Conv2d(n_features, n_features, kernel_size = 5, stride = 1, padding = 2)
		self.linear1 = nn.Linear(112*112*n_features, 100)
		self.linear2 = nn.Linear(100, 3)
		#x, y, probability
	
	def forward(self, x):
		fig = plt.figure(figsize = (8, 8))
		columns = 4
		rows = 5
		x = F.relu(self.conv1(x))
		
		#plt.imshow(x[0][0].detach().numpy())
		
		x = F.relu(self.conv2(x))
		for i in range(1, columns*rows +1):
			img = x[0][i].detach().numpy()
			fig.add_subplot(rows, columns, i)
			plt.imshow(img)
		#plt.imshow(self.conv2.weight[0][0].detach().numpy())
		plt.show()
		x = F.relu(self.conv3(x))
		x = x.view(-1, 112*112*n_features)
		x = F.tanh(self.linear1(x))
		#print(x.shape)
		x = F.tanh(self.linear2(x))
		#print(x.shape)
		return x

model = Head_hunter(n_features, input_chanels, batch_size)
def load_mod(checkpoint):
	model.load_state_dict(checkpoint['state_dictionary'])

if load_m:
	model_par = torch.load('models\\model_test.pth.tar')
	model_par_load = model_par.get('state_dictionary')
	model.load_state_dict(model_par_load)

with torch.no_grad():
	output = model(img)
#output[0], output[1], output[2] = output[0]*112, output[1]*112, output[2]*100
print(output*112)