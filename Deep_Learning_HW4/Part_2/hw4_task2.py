'''
Ivan Christian

Homework 4 - Task 2 Transfer Learrning
'''

from loader.dataloader import CustomFlowerDataset

from torchvision import transforms


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

	for index, batch in enumerate(train_loader):
		data = batch['image'].to(device)
		target = batch['label'].long().to(device)

		output = model(data)
		loss = loss_func(output, target)#CrossEntropyLoss
		loss.backward() #calculate the gradient 
		optimizer.step()
		train_losses.append(loss.item())
	train_loss = torch.mean((torch.tensor(train_losses)))

	return train_loss

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



def create_loaders(im_dir, im_paths, label, train_split=0.7, val_split=0.1):

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





def run():
	pass




if __name__=='__main__':
	run()