import numpy as np
import torch
import torch.nn as nn
import set_model_lstm
import htk_io
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.fftpack import dct
from python_speech_features import delta


gpu_dtype = torch.cuda.FloatTensor

eps = 1e-8


layers = 4
hidden_size = 256
num_dirs = 2
stat_file = 'stat_reverb'     # contains input mean and variance

numcep = 13                   # desired number of cepstrums, won't have C0



def read_mv(stat):
   mean_flag = var_flag = False
   m = v = None
   with open(stat) as s:
      for line in s:
         line = line.strip()
         if len(line) < 1: continue
         if "MEAN" in line:
            mean_flag = True
            continue
         if mean_flag:
            m = list(map(float, line.split()))
            mean_flag = False
            continue
         if "VARIANCE" in line:
            var_flag = True
            continue
         if var_flag:
            v = list(map(float, line.split()))
            var_flag = False
            continue
   return np.array(m, dtype = np.float64), np.array(v, dtype = np.float64)



def org_data(utt_feat, skip_frames = 0 ):
   num_frames, num_channels = utt_feat.shape

   if skip_frames > 0:
      utt_feat = np.pad(utt_feat, ((0, skip_frames), (0,0)), mode = 'edge')    # pad the ending frames
      utt_feat = utt_feat[skip_frames:,:]

   return utt_feat.reshape(1, num_frames, num_channels)      # (1, T, D)




def gen_post(feat_list, model, skip_frames):
   model.eval()             # Put the model in test mode (the opposite of model.train(), essentially)

   m, v = read_mv(stat_file)
   if m is None or v is None:
      raise Exception("mean or variance vector does not exist")

   with open(feat_list) as f:
      for line in f:
         line = line.strip()
         if len(line) < 1: continue
         print ("generating features for file", line)
         io = htk_io.fopen(line)
         utt_feat = io.getall()
         utt_feat -= m       # normalize mean
         utt_feat /= (np.sqrt(v) + eps)     # normalize var
         feat_numpy = org_data(utt_feat, skip_frames)
         feat_tensor = torch.from_numpy(feat_numpy).type(gpu_dtype)
         x = Variable(feat_tensor.type(gpu_dtype), volatile = True)
         input_size_list = [x.size(1)]     # number of time steps
         x = nn.utils.rnn.pack_padded_sequence(x, input_size_list, batch_first=True)
         
         out_feat = model(x, input_size_list)

         out_feat_numpy = out_feat.data.cpu().numpy()
         out_feat_numpy = dct(out_feat_numpy, type=2, axis=1, norm='ortho')[:,1:numcep+1]
         out_feat_delta = delta(out_feat_numpy, 2)
         out_feat_ddelta = delta(out_feat_delta, 2)
         out_feat_numpy = np.concatenate((out_feat_numpy, out_feat_delta, out_feat_ddelta), axis = 1)
         out_file = line.replace(".fea", ".mfc")
         io = htk_io.fopen(out_file, mode="wb", veclen = out_feat_numpy.shape[1])
         io.writeall(out_feat_numpy)
         print ("features saved in %s\n" %out_file)



if __name__ == '__main__':
   model_path = 'weights_lstm/your_model_name'
   feat_list = 'all_fbank.lst'
   skip_frames = 5
   model = set_model_lstm.Layered_RNN(rnn_input_size = 31, nb_layers = layers, rnn_hidden_size = hidden_size, bidirectional = True if num_dirs==2 else False, batch_norm = False, num_classes = 31)
   model = model.type(gpu_dtype)
   model.load_state_dict(torch.load(model_path))     # load model params   
   gen_post(feat_list, model, skip_frames)
