'''
Ivan Christian

Homework 4 - Task 2 Transfer Learrning
'''

from loader.dataloader import CustomFlowerDataset


from torchvision import transforms



def custom_no_loading(im_dir, im_paths, label):
	transformation = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])

	dataset = CustomFlowerDataset(im_dir, im_paths, label, transformation = transformation)

	train_set, val_set, test_set = dataset.train_val_test_split(0.7, 0.1)




def train( model, device , train_loader , optimizer, epochs ):
	pass

def validate ( model , device , val_loader ):
	pass

def test (model, device, test_loader):
	pass


def run():
	pass




if __name__=='__main__':
	run()