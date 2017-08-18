import torch
import numpy as np
import os
import glob
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import shutil
import collections
import set_model_lstm
from random import sample
import h5py
import gc



gpu_dtype = torch.cuda.FloatTensor


train_dir_base = 'train_lstm'

train_files = glob.glob(train_dir_base + '/' + '*.h5')
train_num_chunks = len(train_files)

load_chunks_every = 5     # everytime load this many of .h5 data chunks into memory

layers = 4          # number of LSTM layers
hidden_size = 256   # number of cells per direction
num_dirs = 2        # number of LSTM directions


assert num_dirs == 1 or num_dirs == 2, 'num_dirs must be 1 or 2'



def _collate_fn(batch):

   def func(p):
      """
        p is a tuple, p[0] is the input tensor, p[1] is the target tensor
        data_tensor: (T, F)
        input tensor and target tensor have the same temporal dimension
      """
      return p[0].size(0)

   batch = sorted(batch, reverse = True, key = func)
   longest_sample_src = batch[0][0]
   longest_sample_tgt = batch[0][1]
   feat_size_src = longest_sample_src.size(1)
   feat_size_tgt = longest_sample_tgt.size(1)
   minibatch_size = len(batch)
   max_seqlength = longest_sample_src.size(0)
   inputs = torch.zeros(minibatch_size, max_seqlength, feat_size_src)       # (N, Tmax, D)
   targets = torch.zeros(minibatch_size, max_seqlength, feat_size_tgt)
   input_sizes_list = []
   for x in range(minibatch_size):
        sample = batch[x]
        input_tensor = sample[0]
        target_tensor = sample[1]
        seq_length = input_tensor.size(0)
        inputs[x].narrow(0, 0, seq_length).copy_(input_tensor)
        targets[x].narrow(0, 0, seq_length).copy_(target_tensor)
        input_sizes_list.append(seq_length)
        
   return inputs, targets, input_sizes_list




def get_loader(chunk_list, mode):
  class MyDataset(torch.utils.data.Dataset):
      def __init__(self):
          self.data_files = {}
          i = 0
          for f in chunk_list:
             if mode == 'train': print ('Loading data from %s' %f)
             with h5py.File(f, 'r') as hf:
                for grp in hf:
                    self.data_files[i] = (torch.FloatTensor(np.asarray(hf[grp]['data'])), torch.FloatTensor(np.asarray(hf[grp]['label'])))
                    i += 1
          if mode == 'train': print ('Total %d sequences loaded' %len(self.data_files))

      def __getitem__(self, idx):
          return self.data_files[idx]

      def __len__(self):
          return len(self.data_files)

  dset = MyDataset()
  if mode == 'train':
      loader = DataLoader(dset, batch_size = 4, shuffle = True, collate_fn = _collate_fn, num_workers = 10, pin_memory = False)
  elif mode == 'test':
      loader = DataLoader(dset, batch_size = 6, shuffle = False, collate_fn = _collate_fn, num_workers = 10, pin_memory = False)
  else:
      raise Exception('mode can only be train or test')

  return loader





def train_one_epoch(model, loss_fn, optimizer, print_every = 10):
    data_list = sample(train_files, train_num_chunks)
    model.train()
    t = 0

    for i in range(0, train_num_chunks, load_chunks_every):
          chunk_list = data_list[i: i + load_chunks_every]
          loader_train = get_loader(chunk_list, 'train')
          for data in loader_train:
             inputs, targets, input_sizes_list = data
             batch_size = inputs.size(0)
             inputs = Variable(inputs, requires_grad=False).type(gpu_dtype)
             targets = Variable(targets, requires_grad=False).type(gpu_dtype)

             inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_sizes_list, batch_first=True)
             targets = nn.utils.rnn.pack_padded_sequence(targets, input_sizes_list, batch_first=True).data

             out = model(inputs, input_sizes_list)

             loss = loss_fn(out, targets)     # mse loss

             if (t + 1) % print_every == 0:
                 print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))
             optimizer.zero_grad()
             loss.backward()

             torch.nn.utils.clip_grad_norm(model.parameters(), 400)     # clip gradients
             optimizer.step()
             t += 1
          loader_train.dataset.data_files.clear()
          del loader_train
          gc.collect()
    
             


                 
def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay




def train_epochs(model, loss_fn, init_lr, model_dir):
   if os.path.exists(model_dir):
      shutil.rmtree(model_dir)
   os.makedirs(model_dir)
   
   optimizer = optim.Adam(model.parameters(), lr = init_lr)     # setup the optimizer

   learning_rate = init_lr
   max_iter = 8
   start_halfing_iter = 2
   halfing_factor = 0.25

   count = 0
   half_flag = False

   while count < max_iter:
     count += 1
     if count >= start_halfing_iter:
        half_flag = True
     
     print ("Starting epoch", count)


     if half_flag:
        learning_rate *= halfing_factor
        adjust_learning_rate(optimizer, halfing_factor)     # decay learning rate

     model_path = model_dir + '/epoch' + str(count) + '_lr' + str(learning_rate) + '.pkl'
     train_one_epoch(model, loss_fn, optimizer)      # train one epoch
     torch.save(model.state_dict(), model_path)

   print ("End training")





if __name__ == '__main__':

   # rnn_input_size = input feature dimension
   # num_classes = output feature dimension
   model = set_model_lstm.Layered_RNN(rnn_input_size = 31, nb_layers = layers, rnn_hidden_size = hidden_size, bidirectional = True if num_dirs==2 else False, batch_norm = False, num_classes = 31)
   model = model.type(gpu_dtype)
   loss_fn = nn.MSELoss().type(gpu_dtype)

   train_epochs(model, loss_fn, 1e-3, 'weights_lstm')


