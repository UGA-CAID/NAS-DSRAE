import os
import torch
import numpy as np
import scipy.io as sio
from collections import Counter
np.set_printoptions(precision = 2, suppress = True)
from scipy import stats
import random

###add words%%

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter() # count the hashable objects
                                 # hashable objects: have stable hash value
        self.total = 0
        
    def add_word(self, word, idx):    # add a new word and update the index
        #if word not in self.word2idx:
        #if self.total == sub_num*253*59421:
        #self.word2idx = {}
        #self.idx2word = []
            #self.counter = Counter() # count the hashable objects
                                 # hashable objects: have stable hash value
            #self.total = 0
        self.idx2word.append(word)
        #print(len(self.idx2word))
        #print(word)
            #self.word2idx[word] =len(self.idx2word)-1#idx
        self.word2idx[idx] =word
        #print(len(self.idx2word) )
        #print(self.word2idx[idx])
        token_id = len(self.idx2word)-1
        #token_id = self.word2idx[word]
        #token_id = i
        self.counter[token_id] += 1
      
        self.total += 1
        # print('data word2idx')
        # print(self.word2idx[word])
        #return self.word2idx[word]
        #print(self.word2idx[idx])
        return len(self.idx2word)-1

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self):
        global sub_data_train
        global sub_data_valid
        global sub_data_test
        global SUB_NUM
        global len_volume
        global len_voxel
        global sub_data_test
        global sub_data_valid
        global sub_data_train
        global img_step
        self.dictionary = Dictionary()
        #self.idx2word = []
        len_volume = 405
        
        
        subData = sio.loadmat('WM_360ROI')
        subData = subData['sub_data']
        print(subData.shape)
        #subData = sio.loadmat('/storage/darts_rnn/Gambling_sub_norm_700_50')
        #subData = subData['y_norm']
        subData = np.reshape(subData, (791, len_volume,360))
        subData = np.swapaxes(subData,0,2)
        
        print(np.shape(subData))

        sub_train , sub_valid, sub_test = subData[:,:,0:450],  subData[:,:,450:600], subData[:,:,600:750]
        #self.data = self.tokenize(sub_train,100)
        self.train = self.tokenize(sub_train, 450)
        self.valid = self.tokenize(sub_valid, 150)
        self.test = self.tokenize(sub_test, 150)

    def listToString(self,s):
        self.str1 = ""
        return self.str1.join(s)
    

    def tokenize(self, sub_data, sub_num):
        """Tokenizes a text file."""
        # with open(path, 'r', encoding='utf-8') as f:
        #global rand_num
        
        #rand_num = [i for i in range(0, 59421)]
        #random.shuffle(rand_num)

        #sub_data = sub_data[rand_num[0: 128],:, :]  ### the number of rows
        #del rand_num
        
        tokens = 0
        token = 0
        #self.idx2word = []
        ids = torch.LongTensor(sub_num* len_volume)
        for sub in range(0, sub_num):
            for line in range(0, len_volume):
             #   for word in range (0,360):
                
                words_str = []
                words = sub_data[:, line, sub]# every voxel's temporal as a word
                #words = np.str(words)
                #for i in range(0, len(words)):
                #    word = np.str(words[i])
                #    words_str.append(word)
                #words_str = self.listToString(words_str)
                #tokens += len(words_str)#1
                #self.dictionary.add_word(words, tokens)
                #self.idx2word.append(words)
                
                self.dictionary.add_word(words,token)
                tokens += 1#1
                #ids[token] = token
                #token += 1

        #print(tokens)
        ids = torch.LongTensor(tokens)  # LongTensor for index
        token = 0
        for sub in range(0, sub_num):
            for line in range(0, len_volume):
            #for wordid in range(0, len_volume):
                words = sub_data[:,line, sub]
                #words = np.str(words)
                #words_str = []
                #for i in range(0, len(words)):
                #    word = np.str(words[i])
                #    words_str.append(word)
                #words_str = self.listToString(words_str)
                # for word in words:
                #print(self.dictionary.word2idx)
         #       ids[token] = token
                ids[token] = token
                #ids[token] = self.dictionary.word2idx[words]
                token += 1#1
        #print(ids[3]) 
          
        return ids




class SentCorpus(object):
    def __init__(self, sub_data):
        self.dictionary = Dictionary()
        self.train = self.tokenize(sub_data_train)
        self.valid = self.tokenize(sub_data_valid)
        self.test = self.tokenize(sub_data_test)

    def tokenize(self, sub_data, sub_num):
        """Tokenizes a text file."""
        # assert os.path.exists(path)
        # Add words to the dictionary
        # with open(path, 'r', encoding='utf-8') as f:
        tokens = 0
        #sub_data = sub_data[rand_num[0:128],:, : ]
        for sub in range(0, sub_num):
            for line in range(0, len_volume):
            #for wordid in range(0, len_volume):
              words = sub_data[:,line , sub]
              #words = np.str(words)
                # print('data try words')
                # print(np.size(words))
              tokens += 1
            # print('tokens' + str(tokens))
            #     for word in words:
              self.dictionary.add_word(words, tokens)

        # Tokenize file content
        sents = []
        # with open(path, 'r', encoding='utf-8') as f:
        for sub in range(0, sub_num):
           for line in range(0, len_volume):
            #for wordid in range(0, len_volume):
                words = sub_data[:, line, sub]
                words = np.str(words)
                # for word in words:
                sent = torch.LongTensor(np.shape(words)[0])
                sent[line] = self.dictionary.word2idx[words]
                sents.append(sent)


        return sents


class BatchSentLoader(object):
    def __init__(self, sents, batch_size, pad_id=0, cuda=False, volatile=False):
        self.sents = sents
        self.batch_size = batch_size
        self.sort_sents = sorted(sents, key=lambda x: x.size(0))  # sort sents with x_size
        self.cuda = cuda
        self.volatile = volatile
        self.pad_id = pad_id

    def __next__(self):
        if self.idx >= len(self.sort_sents):  # all the words have been used
            raise StopIteration

        batch_size = min(self.batch_size, len(self.sort_sents) - self.idx)
        batch = self.sort_sents[self.idx:self.idx + batch_size]
        max_len = max([s.size(0) for s in batch])
        tensor = torch.LongTensor(max_len, batch_size).fill_(self.pad_id)
        for i in range(len(batch)):
            s = batch[i]
            tensor[:s.size(0), i].copy_(s)  # transfer batch content into tensor
        if self.cuda:
            tensor = tensor.cuda()

        self.idx += batch_size

        return tensor

    next = __next__

    def __iter__(self):
        self.idx = 0
        return self


if __name__ == '__main__':
    corpus = SentCorpus('../gambling')
    loader = BatchSentLoader(corpus.test, 2)
    for i, d in enumerate(loader):
        print('i, d.size')
        print(i, d.size())


