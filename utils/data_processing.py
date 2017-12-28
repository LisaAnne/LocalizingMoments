import numpy as np
import sys
import os
sys.path.append('utils/')
from config import *
from utils import *
sys.path.append(pycaffe_dir)
import time
import pdb
import random
import pickle as pkl
import caffe
from multiprocessing import Pool
from threading import Thread
import random
import h5py
import itertools
import math
import re

glove_dim = 300
glove_path = 'data/glove.6B.%dd.txt' %glove_dim
#glove_path = 'data/glove_debug_path.txt' #for debugging

if glove_path == 'data/glove_debug_path.txt':
    print "continue?"
    pdb.set_trace()
    
possible_segments = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)]
for i in itertools.combinations(range(6), 2):
    possible_segments.append(i)


length_prep_word = 40
length_prep_character = 250

vocab_file = 'data/vocab_glove_complete.txt'

def word_tokenize(s):
  sent = s.lower()
  sent = re.sub('[^A-Za-z0-9\s]+',' ', sent)
  return sent.split()

def sentences_to_words(sentences):
  words = []
  for s in sentences:
    words.extend(word_tokenize(str(s.lower())))
  return words

class glove_embedding(object):

  ''' Creates glove embedding object
  '''

  def __init__(self, glove_file=glove_path):
    glove_txt = open(glove_file).readlines()
    glove_txt = [g.strip() for g in glove_txt]
    glove_vector = [g.split(' ') for g in glove_txt]
    glove_words = [g[0] for g in glove_vector]
    glove_vecs = [g[1:] for g in glove_vector]
    glove_array = np.zeros((glove_dim, len(glove_words)))
    glove_dict = {}
    for i, w in enumerate(glove_words):  glove_dict[w] = i
    for i, vec in enumerate(glove_vecs):
      glove_array[:,i] = np.array(vec)
    self.glove_array = glove_array
    self.glove_dict = glove_dict
    self.glove_words = glove_words

class zero_language_vector(object):

  def __init__(self, data):
    self.dim = glove_dim 

  def get_vector_dim(self):
    return self.dim

  def get_vocab_size(self):
    return 0

  def preprocess(self, data):
    embedding = np.zeros((self.get_vector_dim(),)) 
    for d in data:
      d['language_input'] = embedding
      d['gt'] = (d['gt'][0], d['gt'][1])
    return data

class recurrent_language(object):

  def get_vocab_size(self):
    return len(self.vocab_dict.keys()) 

  def preprocess_sentence(self, words):
    vector_dim = self.get_vector_dim()
    sentence_mat = np.zeros((len(words), vector_dim))
    count_words = 0
    for i, w in enumerate(words):
      try:
        sentence_mat[count_words,:] = self.vocab_dict[w]
        count_words += 1
      except:
        if '<unk>' in self.vocab_dict.keys():
          sentence_mat[count_words,:] = self.vocab_dict['<unk>'] 
          count_words += 1
        else:
          pass
    sentence_mat = sentence_mat[:count_words] 
    return sentence_mat
    
  def preprocess(self, data):

    
    for d in data:
      words = sentences_to_words([d['description']])
      d['language_input'] = self.preprocess(words)
    return data

class recurrent_word(recurrent_language):

  def __init__(self, data):

    self.data = data
    vocab = open(vocab_file).readlines()
    vocab = [v.strip() for v in vocab] 
    if '<unk>' not in vocab: 
      vocab.append('<unk>') 

    vocab_dict = {}
    for i, word in enumerate(vocab):
      vocab_dict[word] = i 
    self.vocab_dict = vocab_dict

  def get_vector_dim(self):
    return 1 

class recurrent_embedding(recurrent_language):

  def read_embedding(self):
    print "Reading glove embedding"
    embedding = glove_embedding(glove_path)
    self.embedding = embedding

  def get_vector_dim(self):
    return glove_dim 

  def __init__(self, data):

    self.read_embedding()
    embedding = self.embedding
    vector_dim = self.get_vector_dim()
    self.data = data

    self.data = data
    vocab = open(vocab_file).readlines()
    vocab = [v.strip() for v in vocab] 
    if '<unk>' in vocab: 
      vocab.remove('<unk>') #don't have an <unk> vector.  Alternatively, could map to random vector...
    vocab_dict = {}

    for i, word in enumerate(vocab):
      try:
        vocab_dict[word] = embedding.glove_array[:,embedding.glove_dict[word]] 
      except:
        print "%s not in glove embedding" %word
    self.vocab_dict = vocab_dict

  def preprocess(self, data):

    vector_dim = self.get_vector_dim()
    
    for d in data:
      d['language_input'] = sentences_to_words([d['description']])

    return data
  
  def get_vocab_dict(self):
    return self.vocab_dict

#Methods for extracting visual features

def feature_process_base(start, end, features):
  return np.mean(features[start:end+1,:], axis = 0)

def feature_process_norm(start, end, features):
  base_feature = np.mean(features[start:end+1,:], axis = 0)
  return base_feature/(np.linalg.norm(base_feature) + 0.00001)

def feature_process_context(start, end, features):
  feature_dim = features.shape[1]
  full_feature = np.zeros((feature_dim*2,))
  if np.sum(features[5,:]) > 0:
    full_feature[:feature_dim] = feature_process_norm(0,6, features) 
  else:
    full_feature[:feature_dim] = feature_process_norm(0,5, features) 
  full_feature[feature_dim:feature_dim*2] = feature_process_norm(start, end, features) 

  return full_feature

feature_process_dict = {'feature_process_base': feature_process_base, 
                        'feature_process_norm': feature_process_norm,
                        'feature_process_context': feature_process_context,
                        }
   
class extractData(object):
  """ General class to extract data.   
  """

  def increment(self): 
  #uses iteration, batch_size, data_list, and num_data to extract next batch identifiers
    next_batch = [None]*self.batch_size
    if self.iteration + self.batch_size >= self.num_data:
      next_batch[:self.num_data-self.iteration] = self.data_list[self.iteration:]
      next_batch[self.num_data-self.iteration:] = self.data_list[:self.batch_size -(self.num_data-self.iteration)]
      random.shuffle(self.data_list)
      self.iteration = self.num_data - self.iteration
    else:
      next_batch = self.data_list[self.iteration:self.iteration+self.batch_size]
      self.iteration += self.batch_size
    assert self.iteration > -1
    assert len(next_batch) == self.batch_size 
    return next_batch

class extractLanguageFeatures(extractData):

  def __init__(self, dataset, params, result=None):
    self.data_list = range(len(dataset))
    self.num_data = len(self.data_list)
    self.dataset = dataset
    self.iteration = 0

    self.vocab_dict = params['vocab_dict']
    self.batch_size = params['batch_size']
    self.num_glove_centroids = self.vocab_dict.values()[0].shape[0] 
    self.T = params['sentence_length']

    if isinstance(result, dict):
        self.result = result
        self.query_key = params['query_key']
        self.cont_key = params['cont_key']
    
        self.top_keys = [self.query_key, self.cont_key]
        self.top_shapes = [(self.T, self.batch_size, self.num_glove_centroids),
                           (self.T, self.batch_size)]
    else:
        print "Will only be able to run in test mode"

  def get_features(self, query):

    feature = np.zeros((self.T, self.num_glove_centroids)) 
    cont = np.zeros((self.T,)) 

    len_query = min(len(query), self.T)
    if len_query < len(query):
      query = query[:len_query]
    for count_word, word in enumerate(query):
      try:
        feature[-(len_query)+count_word,:] = self.vocab_dict[word] 
      except:
        feature[-(len_query)+count_word,:] = np.zeros((glove_dim,))
    cont[-(len_query-1):] = 1 
    assert np.sum(feature[:-len_query,:]) == 0

    return feature, cont

  def get_data_test(self, data):
    query = data['language_input']
    return self.get_features(query) 

  def get_data(self, next_batch):

    data = self.dataset
    query_mat = np.zeros((self.T, self.batch_size, self.num_glove_centroids))
    cont = np.zeros((self.T, self.batch_size))

    for i, nb in enumerate(next_batch):
      query = data[nb]['language_input']
      query_mat[:,i,:], cont[:,i] = self.get_features(query)

    self.result[self.query_key] = query_mat
    self.result[self.cont_key] = cont 

class extractVisualFeatures(extractData):
  
  def __init__(self, dataset, params, result):
    self.data_list = range(len(dataset))
    self.feature_process_algo = params['feature_process']
    self.loc_feature = params['loc_feature']
    self.num_data = len(self.data_list)
    self.dataset = dataset
    self.iteration = 0
    self.loc = params['loc_feature']
    loss_type = params['loss_type']
    assert loss_type in ['triplet', 'inter', 'intra']

    self.inter = False
    self.intra = False
    if loss_type in ['triplet', 'inter']:
      self.inter = True
    if loss_type in ['triplet', 'intra']:
      self.intra = True

    self.batch_size = params['batch_size']
    self.num_glove_centroids = params['num_glove_centroids']

    features_h5py = h5py.File(params['features'])
    features = {}
    for key in features_h5py.keys():
      features[key] = np.array(features_h5py[key])
    features_h5py.close()
    self.features = features

    assert self.feature_process_algo in feature_process_dict.keys()
    self.feature_process = feature_process_dict[self.feature_process_algo]

    self.feature_dim = self.feature_process(0,0,self.features[self.dataset[0]['video']]).shape[-1]
    self.result = result

    self.feature_key_p = params['feature_key_p']
    self.feature_time_stamp_p = params['feature_time_stamp_p']
    self.feature_time_stamp_n = params['feature_time_stamp_n']

    self.top_keys = [self.feature_key_p, self.feature_time_stamp_p, self.feature_time_stamp_n]
    self.top_shapes = [(self.batch_size, self.feature_dim),
                     (self.batch_size, 2),
                     (self.batch_size,2)]

    if self.inter:
      self.feature_key_inter = 'features_inter'
      self.top_keys.append(self.feature_key_inter)
      self.top_shapes.append((self.batch_size, self.feature_dim)) 
    if self.intra:
      self.feature_key_intra = 'features_intra'
      self.top_keys.append(self.feature_key_intra)
      self.top_shapes.append((self.batch_size, self.feature_dim)) 
    
    self.possible_annotations = possible_segments 

  def get_data_test(self, d):
      video_feats = self.features[d['video']]
      features = np.zeros((len(self.possible_annotations), self.feature_dim))
      loc_feats = np.zeros((len(self.possible_annotations), 2))
      for i, p in enumerate(self.possible_annotations):
          features[i,:] = self.feature_process(p[0], p[1], video_feats)
          loc_feats[i,:] = [p[0]/6., p[1]/6.]

      return features, loc_feats

  def get_data(self, next_batch):

    feature_process = self.feature_process
    data = self.dataset
    features_p = np.zeros((self.batch_size, self.feature_dim))
    if self.inter: features_inter = np.zeros((self.batch_size, self.feature_dim))
    if self.intra: features_intra = np.zeros((self.batch_size, self.feature_dim))

    features_time_stamp_p = np.zeros((self.batch_size, 2))
    features_time_stamp_n = np.zeros((self.batch_size, 2))

    for i, nb in enumerate(next_batch):
        rint = random.randint(0,len(data[nb]['times'])-1)
        gt_s = data[nb]['times'][rint][0]
        gt_e = data[nb]['times'][rint][1]
        possible_n = list(set(self.possible_annotations) - set(((gt_s,gt_e),))) 
        random.shuffle(possible_n)
        n = possible_n[0]
        assert n != (gt_s, gt_e) 
   
        video = data[nb]['video']
        feats = self.features[video]
  
        if self.inter:
          other_video = data[nb]['video']
          while (other_video == video):
            other_video_index = int(random.random()*len(data))
            other_video = data[other_video_index]['video'] 
          feats_inter = self.features[other_video]
   
        features_p[i,:] = feature_process(gt_s, gt_e, feats)
        if self.intra:
          features_intra[i,:] = feature_process(n[0], n[1], feats)
        if self.inter:
          try:
            features_inter[i,:] = feature_process(gt_s, gt_e, feats_inter)
          except:
            pdb.set_trace() 
  
        if self.loc:
          features_time_stamp_p[i,0] = gt_s/6.
          features_time_stamp_p[i,1] = gt_e/6.
          features_time_stamp_n[i,0] = n[0]/6.
          features_time_stamp_n[i,1] = n[1]/6.
        else:
          features_time_stamp_p[i,0] = 0 
          features_time_stamp_p[i,1] = 0
          features_time_stamp_n[i,0] = 0
          features_time_stamp_n[i,1] = 0
  
        assert not math.isnan(np.mean(self.features[data[nb]['video']][n[0]:n[1]+1,:]))
        assert not math.isnan(np.mean(self.features[data[nb]['video']][gt_s:gt_e+1,:]))

    self.result[self.feature_key_p] = features_p
    self.result[self.feature_time_stamp_p] = features_time_stamp_p
    self.result[self.feature_time_stamp_n] = features_time_stamp_n
    if self.inter:
      self.result[self.feature_key_inter] = features_inter
    if self.intra:
      self.result[self.feature_key_intra] = features_intra

class batchAdvancer(object):
  
  def __init__(self, extractors):
    self.extractors = extractors
    self.increment_extractor = extractors[0]

  def __call__(self):
    #The batch advancer just calls each extractor
    next_batch = self.increment_extractor.increment()
    for e in self.extractors:
      e.get_data(next_batch)

class python_data_layer(caffe.Layer):
  """ General class to extract data.
  """

  def setup(self, bottom, top):
    random.seed(10) 
    self.params = eval(self.param_str)
    params = self.params

    assert 'top_names' in params.keys()

    #set up prefetching
    self.thread_result = {}
    self.thread = None

    self.setup_extractors()
 
    self.batch_advancer = batchAdvancer(self.data_extractors) 
    shape_dict = {}
    self.top_names = []
    for de in self.data_extractors:
      for top_name, top_shape in zip(de.top_keys, de.top_shapes):
        shape_dict[top_name] = top_shape 
        self.top_names.append((params['top_names'].index(top_name), top_name)) 

    self.dispatch_worker()

    self.top_shapes = [shape_dict[tn[1]] for tn in self.top_names]

    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    #for top_index, name in enumerate(self.top_names.keys()):

    top_count = 0
    for top_index, name in self.top_names:
      shape = self.top_shapes[top_count] 
      print 'Top name %s has shape %s.' %(name, shape)
      top[top_index].reshape(*shape)
      top_count += 1

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    if self.thread is not None:
      self.join_worker()

    for top_index, name in self.top_names:
      top[top_index].data[...] = self.thread_result[name]

    self.dispatch_worker()

  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def backward(self, top, propoagate_down, bottom):
    pass

feature_process_dict = {'feature_process_base': feature_process_base,
                        'feature_process_norm': feature_process_norm,
                        'feature_process_context': feature_process_context,
                        }

language_feature_process_dict = {'zero_language': zero_language_vector,
                                 'recurrent_embedding': recurrent_embedding}

class dataLayer_ExtractPairedLanguageVision(python_data_layer):
 
  def setup_extractors(self):
    assert 'top_names' in self.params.keys()
    assert 'descriptions' in self.params.keys()
    assert 'features' in self.params.keys()
    if 'batch_size' not in self.params.keys(): self.params['batch_size'] = 120

    self.params['query_key'] = 'query'
    self.params['feature_key_n'] = 'features_n'
    self.params['feature_key_p'] = 'features_p'
    self.params['feature_key_t'] = 'features_t'
    self.params['feature_time_stamp_p'] = 'features_time_stamp_p'
    self.params['feature_time_stamp_n'] = 'features_time_stamp_n'
    self.params['cont_key'] = 'cont'

    language_extractor_fcn = extractLanguageFeatures
    visual_extractor_fcn = extractVisualFeatures
    language_process = recurrent_embedding 

    data_orig = read_json(self.params['descriptions'])
    random.shuffle(data_orig)
    language_processor = language_process(data_orig)
    data = language_processor.preprocess(data_orig)
    self.params['vocab_dict'] = language_processor.vocab_dict
    num_glove_centroids = language_processor.get_vector_dim()
    self.params['num_glove_centroids'] = num_glove_centroids
    visual_feature_extractor = visual_extractor_fcn(data, self.params, self.thread_result)
    textual_feature_extractor = language_extractor_fcn(data, self.params, self.thread_result)
    self.data_extractors = [visual_feature_extractor, textual_feature_extractor]
