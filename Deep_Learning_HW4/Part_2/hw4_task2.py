'''
Ivan Christian

Homework 4 - Task 2 Transfer Learrning
'''



from torch.utils.data import DataLoader

from torchvision import transforms


from loader.dataloader import CustomFlowerDataset
from utils.vis import visualise_train_val_loss


def create_loaders(im_dir, im_paths, label, train_split=0.7, val_split=0.1):
	'''
	Function create_loaders is a function to create data loader for training

	Input:
	- im_dir : string
	- im_paths : string
	- label : 
	- train_split : float ()
	- val_split : float ()

	Output:
	- train_loader : DataLoader for training 
	- val_loader : DataLoader for validation
	- test_loader : DataLoader for testing
	'''

	transformation = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])

	dataset = CustomFlowerDataset(im_dir, im_paths, label, transformation = transformation)

	train_set, val_set, test_set = dataset.train_val_test_split(train_split, val_split)


	'''
	no need to make sure num workers = default since no lambda function exists here
	'''
	train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
	val_loader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=4)

	return train_loader, val_loader, test_loader


def train( model, device , train_loader , optimizer, epochs, loss_func ):
	'''
	Function train is a function to train the model
	input: 
	- model : model
	- device : device that is being used
	- train_loader : data loader for training
	- optimizer : which optimizer used
	- epoch : at what epoch this is runnning in
	- loss_func : loss function used
	Output:
	- train_loss : float ( average train loss )
	'''

	model.train()

	train_losses = []

	correct = 0

	for index, batch in enumerate(train_loader):
		data = batch['image'].to(device)
		target = batch['label'].long().to(device)

		output = model(data)
		loss = loss_func(output, target)#CrossEntropyLoss
		loss.backward() #calculate the gradient 
		optimizer.step()


		# correct += (output == target).float().sum()


		train_losses.append(loss.item())
	train_loss = torch.mean((torch.tensor(train_losses)))

	return train_loss, train_accuracy

def validate ( model , device , val_loader , loss_func):
	'''
	Function validate is used to check accuracy against the validation set
	
	Input:
	- model
	- device
	- val_loader
	- loss_func

	Output:
	- val_loss
	- accuracy
	'''

	model.eval()
	val_loss = 0
	accuracy = 0

	with torch.no_grad():

		for _, batch in enumerate(val_loader):
			data = batch['image'].to(device)
			target = batch['label'].long().to(device)

			batch_loss = loss_func(output, target).item()#CrossEntropyLoss
			val_loss += batch_loss

			pred = output.argmax(dim=1, keepdim=True)
			accuracy += pred.eq(target.view_as(pred)).sum().item()
	val_loss /= len(val_loader)

	accuracy /= len(val_loader.dataset)
	return val_loss, accuracy

def test (model, device, test_loader):
	'''
	Function test is to test the model against the test set
	'''
	model.eval()
	accuracy = 0
	with torch.no_grad():

		for _, batch in enumerate(val_loader):
			data = batch['image'].to(device)
			labels = batch['label'].long().to(device)

			output = model(data)
			pred = result.argmax(dim=1, keepdim= True)
			accuracy += pred.eq(labels.view_as(pred)).sum().item()

	accuracy /= len(test_loader.dataset)
	return accuracy 


def custom_training(im_dir, im_paths, label, mode = 1, epochs = 10):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if mode == 1:
		# mode 1 is to train from scracth
		model = models.resnet18(num_classes=102).to(device)
	elif mode == 2:
		# mode 2 is to do training all layers
		model = models.resnet18(pretrained=True).to(device)
	elif model == 3:
		# mode 3 is to do pre trained weights
		model = models.resnet18(pretrained=True).to(device)
		model.fc = nn.Linear(512, 102)

	else:
		print('Mode not recognised')
		return 

	optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

	loss_func = nn.CrossEntropyLoss()

	train_loader, val_loader, test_loader = create_loaders(im_dir, im_paths, label)


	train_losses_vis = []
	val_losses_vis = []

	for epoch in range(1, epochs + 1):
		train_loss = train(model, device, train_loader, optimizer, epoch)
		val_loss, val_accuracy = validate(model, device, val_loader)

		if (len(val_losses) > 0) and (val_loss < min(val_losses)):
			torch.save(model.state_dict(), "best_model.pt")
		train_losses.append(train_loss)
    	val_losses.append(val_loss)
	
	print('Finished Training and Validation Process')

	test_accuracy = 

	if mode == 1:
		visualise_train_val_loss(train_losses_vis, val_losses_vis, epochs, 'custom_full_training.png' )
	elif mode == 2:
		visualise_train_val_loss(train_losses_vis, val_losses_vis, epochs, 'custom_last2_layers.png' )
	elif model == 3:
		visualise_train_val_loss(train_losses_vis, val_losses_vis, epochs, 'custom_pre_trained.png' )

def run():
	pass


if __name__=='__main__':
	run()