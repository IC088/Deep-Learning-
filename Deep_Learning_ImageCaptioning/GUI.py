'''
GUI code for the Image Captioning visualisation 

Gong Chen           1002870
Ivan Christian      1003056
Lim Theck Sean      1002777
Tang Mingzheng Paul 1002768
'''


from tkinter import *
import os
from PIL import Image, ImageTk
from torchvision import transforms
from utils.model import Generator
import numpy as np
import torch
from utils.data_loader import get_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.initmodels()
        self.getandset_imagecaptionpair()
        
        self.pack(fill=BOTH, expand='yes', side="bottom")
        self.img = Label(self)
        self.setimage()
        self.setlabel()
        self.setbutton()
        
        root.wm_title("Deep Learning")
        root.geometry("750x500")
        root.mainloop()
    
    def initmodels(self):
        transform_train = transforms.Compose([ 
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])
        self.testdata_loader = get_loader(transform=transform_train,
                             mode='test',
                             batch_size=1,
                             vocab_threshold=5,
                             vocab_from_file=True)

        embed_size = 300           # dimensionality of image and word embeddings
        hidden_size = 512          # number of features in hidden state of the RNN decoder
        vocab_size = len(self.testdata_loader.dataset.vocab)
        model = os.path.join('scratch', 'models','final_coop-generator-3-25000.pkl')
        self.Generator = Generator(embed_size, hidden_size, vocab_size, embed_size, num_layers=2)
        self.Generator.load_state_dict(torch.load(model))
        self.encoder = self.Generator.cnn
        self.decoder = self.Generator.rnn
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()
        

    def resizeimg(self, img):
        basewidth = 750
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        return img
    
    #Generates the next image caption pair
    def clicked(self):
        print('Clicked')
        self.getandset_imagecaptionpair()
        self.img.config(image='')
        self.caption.config(text='')
        self.setimage()
        self.setlabel()
    
    def setimage(self):
        load = Image.open(self.imagepath)
        load = self.resizeimg(load)
        render = ImageTk.PhotoImage(load)
        self.img.config(image=render)
        self.img.image = render
        self.img.place(x=0, y=0)
    
    def setlabel(self):
        self.caption= Label(self, text="Lazy Sampling: "+self.imagecaption)
        self.caption.place(x=0,y=450)
    
    def setbutton(self):
        btn = Button(self, text="Next Pair", command=self.clicked)
        btn.place(x=350,y=475)
        
        
    def getandset_imagecaptionpair(self):
        orig_image, image, imagepath, _ , _  = next(iter(self.testdata_loader))
        image = image.to(device)
        features = self.encoder(image).unsqueeze(1)
        output = self.decoder.sample(features, states=None, max_len=15)
        sentence = self.clean_sentence(output)
        
        self.image = orig_image
        # self.imagepath = imagepath[0]
        self.imagepath = './venom_pope.png'
        self.imagecaption = sentence
        
        return orig_image, sentence
    
    def clean_sentence(self, output):

        words_sequence = []

        for i in output:
            if (i == 1):
                continue
            words_sequence.append(self.testdata_loader.dataset.vocab.idx2word[i])

        words_sequence = words_sequence[1:-1] 
        sentence = ' '.join(words_sequence) 
        sentence = sentence.capitalize()

        return sentence



if __name__ == '__main__':
    root = Toplevel()
    app = Window(root)

