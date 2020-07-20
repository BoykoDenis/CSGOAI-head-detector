import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from numpy import loadtxt
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#init
epochs = 30
lr = 0.01
dataset_size = 1746
batch_size = 10
n_features = 100
input_chanels = 1
load_m = False
model_path = "models\\model_test.pth.tar"

dataset_path = "D:\\Datasets\\Head_hunt\\"
training_dataset = []
validation_dataset = []
validation_labels = []
training_labels = []

def save_mod(state, filename = model_path):
	torch.save(state, filename)

def load_mod(checkpoint):
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(['state_dict'])

def im_convert(tensor):
	image = tensor.cpu().clone().detach().numpy()
	#clone tensor --> detach it from computations --> transform to numpy
	#image = image.squeeze()
	#image = image.transpose(1, 2, 0)
	# swap axis from(1,28,28) --> (28,28,1)
	#image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
	#denormalize image
	#image = image.clip(0, 1)
	#sets image range from 0 to 1
	return image

# load array
labels = loadtxt('data.csv', delimiter=',')

# print the array
print("converting labels...", end = "\r")

for idx, i in enumerate(labels):
	if i[2] == -1:
		i[0], i[1], i[2] = -0.9, -0.9, -0.9
	else:
		labels[idx][2] = 112.0 
		labels[idx] = labels[idx]/112.0




print("labels_converted...", end = "\r")

#dataloading
raw_data = [Image.open(dataset_path + "head (" + str(i) + ").png" ).convert("L") for i in range(1, dataset_size)]

transform_train = transforms.Compose([transforms.ToTensor(),
									  transforms.Normalize((0.5,),
									  					   (0.5,))
									  ])

print("converting images...", end = "\r")

for idx, img in enumerate(raw_data):
	if idx % 20 == 0:
		validation_dataset.append(transform_train(img))
		validation_labels.append(labels[idx])
	else:
		training_dataset.append(transform_train(img))
		training_labels.append(labels[idx])



#plt.imshow(im_convert(training_dataset[1][0]))
#plt.show()

raw_data = None
labels = None

training_labels = torch.from_numpy(np.array(training_labels)).double()
validation_labels = torch.from_numpy(np.array(validation_labels)).double()

print("images_converted...", end = "\r")

training_dataset = torch.stack(training_dataset)
validation_dataset = torch.stack(validation_dataset) 
print(validation_dataset.shape)

#print(type(validation_labels))

training_dataset = torch.utils.data.TensorDataset(training_dataset, training_labels)
validation_dataset = torch.utils.data.TensorDataset(validation_dataset, validation_labels) 

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle = False)

#neural_network_init

class Head_hunter(nn.Module):
	def __init__(self, n_features, input_chanels, batch_size):
		super().__init__()
		self.conv1 = nn.Conv2d(input_chanels, n_features, kernel_size = 5, stride = 1, padding = 2)
		self.conv2 = nn.Conv2d(n_features, n_features, kernel_size = 5, stride = 1, padding = 2)
		self.conv3 = nn.Conv2d(n_features, n_features, kernel_size = 5, stride = 1, padding = 2)
		self.dropout1 = nn.Dropout(0.5)
		self.linear1 = nn.Linear(112*112*n_features, 100)
		self.linear2 = nn.Linear(100, 3)
		#x, y, probability
	
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = x.view(-1, 112*112*n_features)
		x = F.tanh(self.linear1(x))
		#print(x)
		#print(x.shape)
		x = self.dropout1(x)  
		x = F.tanh(self.linear2(x))
		#x[:, 2] = (x[:, 2]*2) - 1
		#print(x)
		return x

print("initializing_model...", end = "\r")

#criterion = nn.BCEWithLogitsLoss() 
criterion = nn.MSELoss(reduce = "sum")
model = Head_hunter(n_features, input_chanels, batch_size).to(device)
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr = lr)

print("starting_training...", end = "\r")

if load_m:
	load_mod(torch.load(model_path))


for epoch in range(epochs):
	running_loss = 0.0
	checkpoint = {'state_dictionary' : model.state_dict(), 'optimizer': optimizer.state_dict()}
	if epoch % 5 == 0:
		save_mod(checkpoint)
	for idx, [image, label] in enumerate(training_loader):
		#print(image.shape)

		image = image.to(device)
		label = label.to(device)

		output = model(image).double().to(device)

		loss = criterion(output, label)

		running_loss += loss.item()
		print("epoch: ", epoch, " iter: ", idx, " loss: ", running_loss/(idx+1), end = "\r")
		
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

	if epoch % 5 == 0:
		validation_loss = 0.0
		with torch.no_grad():
			for idx, [val_img, label] in enumerate(validation_loader):

				val_img = val_img.to(device)
				#print(val_img.shape)
				label = label.to(device)

				output = model(val_img)
				loss = criterion(output, label)
				validation_loss += loss.item()

			print("validation_iter: ", idx, " validation_loss: ", validation_loss/(idx+1))


img = Image.open(dataset_path + "head (" + str(182) + ").png" ).convert("L")
img = transform_train(img)
img = img.unsqueeze(0)
plt.imshow(img[0][0])
plt.show()

output = model(image).double().to(device)
print(output*112)		
		

