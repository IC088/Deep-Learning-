'''
Deep Learning Image Captioning 

Gong Chen           1002870
Ivan Christian      1003056
Lim Theck Sean      1002777
Tang Mingzheng Paul 1002768
'''

import torch
import torch.nn as nn
import torchvision.models as models
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import math
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


from utils.data_loader import get_loader
from utils.model import Generator, Discriminator, EncoderCNN, DecoderRNN, EncoderRNN


def get_training_loader(batch_size, vocab_threshold, vocab_from_file):
    '''
    Creates a data loader for the training/validation process

    Args:
    - mode : string ( 'train')
    - batch_size : int ( batch )
    - vocab_threshold : int 
    - vocab_from_file : bool

    Return:
    - data_loader : DataLoader object (Data loader object for training/testing)
    '''
    transform_train = transforms.Compose([ 
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)
    return data_loader

def train( 
    data_loader, 
    embed_size, 
    hidden_size, 
    vocab_size, 
    batch_size, 
    num_epochs, 
    save_every,
    print_every, 
    total_step, 
    device ):

    generator = Generator(embed_size, hidden_size, vocab_size, embed_size)
    discriminator = Discriminator(embed_size, hidden_size, vocab_size, embed_size)

    gen_params = list(generator.parameters())
    disc_params = list(discriminator.parameters())
    params = generator.train_params + discriminator.train_params

    optimizer = torch.optim.Adam(params = params, lr = 0.0001)


    # move to cuda
    generator.to(device)
    discriminator.to(device)

    #Cosine Similarity
    cosineSim = nn.CosineSimilarity(dim=0, eps=1e-6)
    # Generator Criterion
    criterion_A = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Discriminator Criterion
    pos_weight = torch.Tensor([batch_size-1])

    criterion_B = nn.BCEWithLogitsLoss(pos_weight).cuda()


    # Start Training

    generator.train()
    discriminator.train()


    for epoch in range(1, num_epochs+1):
        
        for i_step in range(1, total_step+1):
            
            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler
            
            # Obtain the batch.
            images, captions = next(iter(data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)
            
            # Zero the gradients.
            generator.zero_grad()
            discriminator.zero_grad()
            
            # Pass the inputs through the CNN-RNN model.
            out, img_feats = generator(images, captions)
            
            rho = 0.25
            if random.random() < rho:
                gen_captions = nn.functional.softmax(out)
            else:
                gen_captions = nn.functional.gumbel_softmax(out, hard=True)
                    
            cap_feats = discriminator(img_feats, gen_captions).to(device)
            
            max_loss_1 = 0
            max_loss_2 = 0
            pos_pair_term = cosineSim(cap_feats[0], img_feats[0])
            for i in range(1,batch_size):
                max_loss_1 += max(0, 1 - pos_pair_term + cosineSim(cap_feats[0], img_feats[i]))
                max_loss_2 += max(0, 1 - pos_pair_term + cosineSim(cap_feats[i], img_feats[0]))
            
            
            
            # Calculate the batch loss.
            loss_A = criterion_A(out.view(-1, vocab_size), captions.view(-1))
            loss_B = (max_loss_1 + max_loss_2)/(batch_size-1)
            
            B_weight = 0.5
            loss = loss_A + B_weight*loss_B
            
            # Backward pass.
            loss.backward()
            
            # Update the parameters in the optimizer.
            optimizer.step()
                
            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: (%.4f, %.4f, %.4f), Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), loss_A.item(), B_weight*loss_B.item(), np.exp(loss.item()))
            
            # Print training statistics (on same line).
            print('\r' + stats, end="")
            
            # Print training statistics (on different line).
            if i_step % print_every == 0:
                print('\r' + stats)
                
            # Save the weights.
            if i_step % save_every == 0:
                torch.save(generator.state_dict(), os.path.join('scratch', 'models',f'coop-generator-{epoch}-{i_step}.pkl'))
                torch.save(discriminator.state_dict(), os.path.join('scratch', 'models',f'coop-discriminator-{epoch}-{i_step}.pkl'))

    print('Training Finished')


def clean_sentence(output, data_loader):
    '''
    Function to convert the predicted indeces to words


    Args:
    - output : list ( list of predicted index outputs )

    returns:
    - sentence : string ( converted predicted sentence )
    '''
    
    words_sequence = []
    
    for i in output:
        if (i == 1):
            continue
        words_sequence.append(data_loader.dataset.vocab.idx2word[i])
    
    words_sequence = words_sequence[1:-1] 
    sentence = ' '.join(words_sequence) 
    sentence = sentence.capitalize()
    
    return sentence


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def get_testing_loader():
    transform_test = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406),
                                                          (0.229, 0.224, 0.225))
                                    ])
    data_loader = get_loader(transform=transform_test,
                         mode='test')
    return data_loader

def test(data_loader, weights,embed_size, hidden_size, vocab_size, device):
    '''

    '''
    orig_image, image , img_path, caption ,img_id = next(iter(data_loader))

    generator = Generator(embed_size, hidden_size, vocab_size, embed_size, num_layers=2).cpu()

    generator.load_state_dict(torch.load(weights))
    generator.eval()

    encoder = generator.cnn
    decoder = generator.rnn

    # encoder.to(device)
    # decoder.to(device)

    features = encoder(image).unsqueeze(1)

    output = decoder.sample(features)

    sentence = clean_sentence(output,data_loader)


    print(sentence.split('.')[0])

    sentence = sentence.split('.')[0] + '. '


    ref = {img_id : [caption[0]]}
    hypo = {img_id : [sentence]}

    scores = score(ref,hypo)



    return img_id, caption, sentence, scores





def run():

    # Hyperparameters and other variables

    batch_size = 8        # batch size
    vocab_threshold = 5        # minimum word count threshold
    vocab_from_file = True # if True, load existing vocab file ,change to False if you want to recreate vocab.pkl
    embed_size = 300   #original is 300        # dimensionality of image and word embeddings
    hidden_size = 512  # original is 512        # number of features in hidden state of the RNN decoder
    num_epochs = 3             # number of training epochs
    save_every = 500             # determines frequency of saving model weights
    print_every = 100         # determines window for printing average loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the data loader

    data_loader = get_training_loader( batch_size, vocab_threshold, vocab_from_file)
    total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)
    vocab_size = len(data_loader.dataset.vocab)



    model = os.path.join('scratch', 'models','final_coop-generator-3-25000.pkl')
    # Start training

    # Check if last model exists, no point in retraining.
    if not os.path.exists(model):
        train( 
            data_loader, 
            embed_size, 
            hidden_size, 
            vocab_size, 
            batch_size, 
            num_epochs, 
            save_every, 
            print_every, 
            total_step, 
            device )


    # Test


    test_loader = get_testing_loader()

    # test_length = len(test_loader)
    test_length = 100

    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    bleu4 = 0
    rougel = 0 

    for i in range(test_length):
        img_id, caption, sentence, scores = test(test_loader, model,embed_size, hidden_size, vocab_size, device)

        print(caption)
        print(sentence)
        bleu1 += scores['Bleu_1']
        bleu2 += scores['Bleu_2']
        bleu3 += scores['Bleu_3']
        bleu4 += scores['Bleu_4']
        rougel += scores['ROUGE_L']


    print(bleu1/ test_length)
    print(bleu2/ test_length)
    print(bleu3/ test_length)
    print(bleu4/ test_length)
    print(rougel/ test_length)


if __name__ == '__main__':
    run()