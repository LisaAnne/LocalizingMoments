from __future__ import print_function
import sys
sys.path.append('utils/')
from config import *
from utils import *
sys.path.append(pycaffe_dir)
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import pdb
import os
import argparse
from data_processing import *
import h5py
import numpy as np
caffe.set_mode_gpu()

class retrieval_net(object):

  def euclidean_distance(self, vec1, vec2, axis=1):
    negative = L.Power(vec2, scale=-1)
    difference = L.Eltwise(vec1, negative, operation=1)
    squared = L.Power(difference, power=2)
    reduction = L.Reduction(squared, axis=axis)
    return reduction
  
  def eltwise_distance(self, vec1, vec2):
    mult = L.Eltwise(vec1, vec2, operation=0)
    norm_mult = self.normalize(mult, numtiles=self.visual_embedding_dim[-1])   

    score = L.InnerProduct(norm_mult, num_output=1, 
                           weight_filler=self.uniform_weight_filler(-0.08, .08), 
                           param=self.learning_params([[1,1], [2, 0]], ['eltwise_dist', 'eltwise_dist_b'])) 

    return score
  
  
  def bilinear_distance(self, vec1, vec2):
      reshape_vec1 = L.Reshape(vec1, shape=dict(dim=[self.batch_size, -1, 1, 1]))
      reshape_vec2 = L.Reshape(vec2, shape=dict(dim=[self.batch_size, -1, 1, 1]))
      bilinear = L.CompactBilinear(reshape_vec1, reshape_vec2)
      signed = L.SignedSqrt(bilinear)
      l2_normalize = L.L2Normalize(signed)
      score = L.InnerProduct(l2_normalize, num_output=1, 
                             weight_filler=self.uniform_weight_filler(-0.08, .08), 
                             param=self.learning_params([[1,1], [2, 0]], ['bilinear_dist', 'bilinear_dist_b'])) 
  
      return score

  def dot_product_distance(self, vec1, vec2, axis=1):
    mult = L.Eltwise(vec1, vec2, operation=0)
    reduction = L.Reduction(mult, axis=axis)
    negative = L.Power(reduction, scale=-1, shift=1)
    return negative

  def __init__(self, args,
               data_layer='dataLayer_ExtractPairedLanguageVision', top_size=5,
               param_str = None, params={},
               is_test=False):

    self.n = caffe.NetSpec()
    self.silence_count = 0
    self.margin = args.margin
    self.is_test = is_test
    self.dropout_visual = args.dropout_visual
    self.dropout_language = args.dropout_language
    self.visual_embedding_dim = args.visual_embedding_dim 
    self.language_embedding_dim = args.language_embedding_dim 
    self.vision_layers = args.vision_layers
    self.language_layers = args.language_layers
    self.loc = args.loc
    self.data_layer = data_layer
    self.top_size = top_size
    self.param_str = param_str
    self.lw_inter = args.lw_inter
    self.lw_intra = args.lw_intra
    self.top_name_dict = params['top_names_dict']
    self.args = args
    self.T = params['sentence_length']
    self.count_im = 0
    self.local_unary_count = 0
    self.global_unary_count = 0

    self.inter = False
    self.intra = False
    if args.loss_type in ['triplet', 'inter']:
      self.inter = True
    if args.loss_type in ['triplet', 'intra']:
      self.intra = True

    assert self.inter or self.intra #need to have some type of loss!

    if 'batch_size' in param_str.keys():
      self.batch_size = param_str['batch_size']
    else:
      self.batch_size =120 

    self.params = params
    self.image_tag = args.image_tag

    if args.distance_function == 'dot_product_distance':
      self.distance_function = self.dot_product_distance
    elif args.distance_function == 'eltwise_distance':
      self.distance_function = self.eltwise_distance
    elif args.distance_function == 'bilinear_distance':
      self.distance_function = self.bilinear_distance
    else:
      self.distance_function = self.euclidean_distance 

  #Network operations I use frequently

  def uniform_weight_filler(self, min_value, max_value):
    return dict(type='uniform', min=min_value, max=max_value)

  def constant_filler(self, value=0):
    return dict(type='constant', value=value)

  def learning_params(self, param_list, name_list = None):
    param_dicts = []
    for il, pl in enumerate(param_list):
      param_dict = {}
      param_dict['lr_mult'] = pl[0]
      if name_list:
        param_dict['name'] = name_list[il]
      if len(pl) > 1:
        param_dict['decay_mult'] = pl[1]
      param_dicts.append(param_dict)
    return param_dicts

  #"layers" needed for localization
  def sum(self, bottoms):
    return L.Eltwise(*bottoms, operation=1)

  def prod(self, bottoms):
    return L.Eltwise(*bottoms, operation=0)

  def rename_tops(self, tops, names):
     if not isinstance(tops, list):
       tops = [tops]
     if isinstance(names, str):
       names = [names]
     for top, name in zip(tops, names): setattr(self.n, name, top)


  def normalize(self, bottom, axis=1, numtiles=4096):
    power = L.Power(bottom, power=2)
    power_sum = L.Reduction(power, axis=axis, operation=1)
    sqrt = L.Power(power_sum, power=-0.5, shift=0.00001)
    if axis == 1:
        reshape = L.Reshape(sqrt, shape=dict(dim=[-1,1])) 
    if axis == 2:
        reshape = L.Reshape(sqrt, shape=dict(dim=[self.batch_size,-1, 1])) 
    tile = L.Tile(reshape, axis=axis, tiles=numtiles) 
    return L.Eltwise(tile, bottom, operation=0)

  def image_model_two_layer(self, bottom, time_stamp=None, axis=1, tag=''):
    if time_stamp: 
        bottom = L.Concat(bottom, time_stamp, axis=1) #time stamp will just be zeros for --no-loc option

    inner_product_1 =  L.InnerProduct(bottom, num_output=self.visual_embedding_dim[0], 
                               weight_filler=self.uniform_weight_filler(-0.08, .08),
                               bias_filler=self.constant_filler(0), 
                               param=self.learning_params([[1,1], [2,0]], ['image_embed1'+tag, 'image_embed_1b'+tag]), axis=axis)

    if self.image_tag:
      setattr(self.n, self.image_tag + 'ip1' + str(self.count_im), inner_product_1)
      self.count_im += 1
    nonlin_1 = L.ReLU(inner_product_1)

    top_visual =  L.InnerProduct(nonlin_1, num_output=self.visual_embedding_dim[1], 
                           weight_filler=self.uniform_weight_filler(-0.08, .08),
                           bias_filler=self.constant_filler(0), 
                           param=self.learning_params([[1,1], [2,0]], ['image_embed2'+tag, 'image_embed_b2'+tag]), axis=axis)

    if self.image_tag:
      setattr(self.n, self.image_tag + 'ip2' + str(self.count_im), top_visual)
      self.count_im += 1
    dropout = L.Dropout(top_visual, dropout_ratio=self.dropout_visual)

    setattr(self.n, 'embedding_visual', dropout)
    return dropout

  def image_model_one_layer(self, bottom, time_stamp=None, axis=1, tag=''):
    if time_stamp: 
        bottom = L.Concat(bottom, time_stamp, axis=1) #time stamp will just be zeros for --no-loc option
    inner_product = L.InnerProduct(bottom, num_output=self.visual_embedding_dim[0], 
                           weight_filler=self.uniform_weight_filler(-0.08, .08),
                           bias_filler=self.constant_filler(0), 
                           param=self.learning_params([[1,1], [2,0]], ['image_embed1'+tag, 'image_embed_1b'+tag]), 
                           axis=axis)
    dropout = L.Dropout(inner_product, dropout_ratio=self.dropout_visual)
    setattr(self.n, 'embedding_visual', dropout)
    return dropout 

  #language_models
  def language_model_lstm_no_embed(self, sent_bottom, cont_bottom, text_name='embedding_text'):

    lstm_lr = self.args.lstm_lr
    embedding_lr = self.args.language_embedding_lr
      
    lstm = L.LSTM(sent_bottom, cont_bottom, 
                  recurrent_param = dict(num_output=self.language_embedding_dim[0],
                  weight_filler=self.uniform_weight_filler(-0.08, 0.08),
                  bias_filler = self.constant_filler(0)),
                  param=self.learning_params([[lstm_lr,lstm_lr], [lstm_lr,lstm_lr], [lstm_lr,lstm_lr]], ['lstm1', 'lstm2', 'lstm3'])) 
    lstm_slices = L.Slice(lstm, slice_point=self.params['sentence_length']-1, axis=0, ntop=2)
    self.n.tops['silence_cell_'+str(self.silence_count)] = L.Silence(lstm_slices[0], ntop=0)
    self.silence_count += 1 
    top_lstm = L.Reshape(lstm_slices[1], shape=dict(dim=[-1, self.language_embedding_dim[0]]))
    top_text =  L.InnerProduct(top_lstm, num_output=self.language_embedding_dim[1], 
                               weight_filler=self.uniform_weight_filler(-0.08, .08),
                               bias_filler=self.constant_filler(0), 
                               param=self.learning_params([[embedding_lr,embedding_lr], [embedding_lr*2,0]], ['lstm_embed1', 'lstm_embed_1b']))
    setattr(self.n, text_name, top_text)
    return top_text

  def ranking_loss(self, p, n, t, lw=1):

    #For ranking used in paper
    distance_p = self.distance_function(p, t)
    distance_n = self.distance_function(n, t)
    negate_distance_n = L.Power(distance_n, scale=-1)
    max_sum = L.Eltwise(distance_p, negate_distance_n, operation=1)
    max_sum_margin = L.Power(max_sum, shift=self.margin)
    max_sum_margin_relu = L.ReLU(max_sum_margin, in_place=False)
    ranking_loss = L.Reduction(max_sum_margin_relu, operation=4, loss_weight=[lw])

    return  ranking_loss
 
  def write_net(self, save_file, top):
    write_proto = top.to_proto()
    #assert not os.path.isfile(save_file)
      
    with open(save_file, 'w') as f:
      print(write_proto, file=f)
    print("Wrote net to: %s." %save_file) 

  def get_models(self):
    if self.vision_layers == '1':
      vision_layer = self.image_model_one_layer
      assert len(self.visual_embedding_dim) == 1 
    elif self.vision_layers == '2':
      vision_layer = self.image_model_two_layer  
      assert len(self.visual_embedding_dim) == 2 
    else:
      raise Exception("No specified vision layer for %s" %self.vision_layers)

    assert self.language_layers == 'lstm_no_embed' #no other language model implemented

    return vision_layer, self.language_model_lstm_no_embed

  def build_retrieval_model(self, param_str, save_tag):

    #TODO:  This would perhaps be cleaner if I did not co-sample inter/intra positives negatives; shouldn't have to do that and could get rid of determining top size...

    #gets all the tops from the data layer, and names them sensible things.
    data = L.Python(module="data_processing", layer=self.data_layer, param_str=str(param_str), ntop=self.top_size)
    for key, value in zip(self.params['top_names_dict'].keys(), self.params['top_names_dict'].values()):
        setattr(self.n, key, data[value])
    
    im_model, lang_model = self.get_models()

    data_bottoms = []

    #bottoms which are always produced
    bottom_positive = data[self.top_name_dict['features_p']]
    query = data[self.top_name_dict['query']]
    p_time_stamp = data[self.top_name_dict['features_time_stamp_p']]
    n_time_stamp = data[self.top_name_dict['features_time_stamp_n']]
    if self.inter:
      bottom_inter = data[self.top_name_dict['features_inter']]
    if self.intra:
      bottom_intra = data[self.top_name_dict['features_intra']]

    bottom_positive = im_model(bottom_positive, p_time_stamp)
    if self.inter:
      bottom_inter = im_model(bottom_inter, p_time_stamp)
    if self.intra:
      bottom_intra = im_model(bottom_intra, n_time_stamp)
    if (self.inter) & (not self.intra):
      self.n.tops['silence_cell_'+str(self.silence_count)] = L.Silence(n_time_stamp, ntop=0)
      self.silence_count += 1      

    cont = data[self.top_name_dict['cont']]
    query = lang_model(query, cont)
    if self.inter:
      self.n.tops['ranking_loss_inter'] = self.ranking_loss(bottom_positive, bottom_inter, query, lw=self.lw_inter)
    if self.intra:
      self.n.tops['ranking_loss_intra'] = self.ranking_loss(bottom_positive, bottom_intra, query, lw=self.lw_intra)
    self.write_net(save_tag, self.n)

  def build_retrieval_model_deploy(self, save_tag, visual_feature_dim, language_feature_dim):

    image_input =  L.DummyData(shape=[dict(dim=[21, visual_feature_dim])], ntop=1) 
    setattr(self.n, 'image_data', image_input) 

    loc_input =  L.DummyData(shape=[dict(dim=[21, 2])], ntop=1) 
    setattr(self.n, 'loc_data', loc_input) 
   
    im_model, lang_model = self.get_models()

    bottom_visual = im_model(image_input, loc_input)

    text_input =  L.DummyData(shape=[dict(dim=[self.params['sentence_length'], 21, language_feature_dim])], ntop=1) 
    setattr(self.n, 'text_data', text_input)  
    cont_input =  L.DummyData(shape=[dict(dim=[self.params['sentence_length'], 21])], ntop=1) 
    setattr(self.n, 'cont_data', cont_input)  
    bottom_text = lang_model(text_input, cont_input)

    self.n.tops['rank_score'] = self.distance_function(bottom_visual, bottom_text)
    self.write_net(save_tag, self.n)

def add_dict_values(key, my_dict):
  if my_dict.values():
    max_value = max(my_dict.values())
    my_dict[key] = max_value + 1
  else:
    my_dict[key] = 0
  return my_dict

def make_solver(save_name, snapshot_prefix, train_nets, test_nets, **kwargs):

  #set default values
  parameter_dict = kwargs
  if 'test_iter' not in parameter_dict.keys(): parameter_dict['test_iter'] = 10
  if 'test_interval' not in parameter_dict.keys(): parameter_dict['test_interval'] = 100
  if 'base_lr' not in parameter_dict.keys(): parameter_dict['base_lr'] = 0.1
  if 'lr_policy' not in parameter_dict.keys(): parameter_dict['lr_policy'] = '"step"' 
  if 'display' not in parameter_dict.keys(): parameter_dict['display'] = 100 
  if 'max_iter' not in parameter_dict.keys(): parameter_dict['max_iter'] = 10000
  if 'gamma' not in parameter_dict.keys(): parameter_dict['gamma'] = 0.1
  if 'stepsize' not in parameter_dict.keys(): parameter_dict['stepsize'] = 5000
  if 'snapshot' not in parameter_dict.keys(): parameter_dict['snapshot'] = 2500
  if 'momentum' not in parameter_dict.keys(): parameter_dict['momentum'] = 0.9
  if 'weight_decay' not in parameter_dict.keys(): parameter_dict['weight_decay'] = 0.0
  if 'solver_mode' not in parameter_dict.keys(): parameter_dict['solver_mode'] = 'GPU'
  if 'random_seed' not in parameter_dict.keys(): parameter_dict['random_seed'] = 1701
  if 'average_loss' not in parameter_dict.keys(): parameter_dict['average_loss'] = 100
  if 'clip_gradients' not in parameter_dict.keys(): parameter_dict['clip_gradients'] = 10
  if 'device_id' not in parameter_dict.keys(): parameter_dict['device_id'] = 0 
  if 'debug_info' not in parameter_dict.keys(): parameter_dict['debug_info'] = 'false'

  if parameter_dict['type'] == '"Adam"':
      parameter_dict['lr_policy'] = '"fixed"' 
      parameter_dict['momentum2'] = 0.999
      parameter_dict['regularization_type'] = '"L2"'
      if 'type' not in parameter_dict.keys(): parameter_dict['delta'] = 0.0000001 

  snapshot_prefix = 'snapshots/%s' %snapshot_prefix
  parameter_dict['snapshot_prefix'] = '"%s"' %snapshot_prefix
 
  write_txt = open(save_name, 'w')
  write_txt.writelines('train_net: "%s"\n' %train_nets)
  for tn in test_nets:
    write_txt.writelines('test_net: "%s"\n' %tn)
    write_txt.writelines('test_iter: %d\n' %parameter_dict['test_iter'])
  if len(test_nets) > 0:
    write_txt.writelines('test_interval: %d\n' %parameter_dict['test_interval'])

  parameter_dict.pop('test_iter')
  parameter_dict.pop('test_interval')

  for key in parameter_dict.keys():
    write_txt.writelines('%s: %s\n' %(key, parameter_dict[key]))
  write_txt.close()
  print("Wrote solver to %s." %save_name)

def train_model(solver_path, net=None):
  solver = caffe.get_solver(solver_path)
  if net:
    solver.net.copy_from(net)
    print("Copying weights from %s" %net)
  solver.solve()
 
if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  #how to tag built nets/snapshots etc.
  parser.add_argument("--tag", type=str, default='') 

  #training data
  parser.add_argument("--train_json", type=str, default='data/train_data.json') 
  parser.add_argument("--train_h5", type=str, default='data/average_fc7.h5') 
  parser.add_argument("--test_json", type=str, default='data/val_data.json') 
  parser.add_argument("--test_h5", type=str, default='data/average_fc7.h5') 

  #net specifications
  parser.add_argument("--feature_process_visual", type=str, default='feature_process_norm') 
  parser.add_argument("--feature_process_language", type=str, default='recurrent_embedding') 
  parser.add_argument('--loc', dest='loc', action='store_true')
  parser.add_argument('--no-loc', dest='loc', action='store_false')
  parser.set_defaults(loc=False)
  parser.add_argument('--loss_type', type=str, default='triplet')
  parser.add_argument('--margin', type=float, default=0.1)
  parser.add_argument('--dropout_visual', type=float, default=0.0)
  parser.add_argument('--dropout_language', type=float, default=0.0)
  parser.add_argument('--visual_embedding_dim', type=int, nargs='+', default=[100])
  parser.add_argument('--language_embedding_dim', type=int, nargs='+', default=[1000, 100])
  parser.add_argument('--lw_inter', type=float, default=0.5)
  parser.add_argument('--lw_intra', type=float, default=0.5)
  parser.add_argument('--vision_layers', type=str, default='1')
  parser.add_argument('--language_layers', type=str, default='lstm_no_embed')
  parser.add_argument('--distance_function', type=str, default='euclidean_distance')
  parser.add_argument('--image_tag', type=str, default=None)

  #learning params
  parser.add_argument('--random_seed', type=int, default='1701')
  parser.add_argument('--max_iter', type=int, default=10000)
  parser.add_argument('--snapshot', type=int, default=5000)
  parser.add_argument('--stepsize', type=int, default=5000)
  parser.add_argument('--base_lr', type=float, default=0.01)
  parser.add_argument('--lstm_lr', type=float, default=10)
  parser.add_argument('--language_embedding_lr', type=float, default=1)
  parser.add_argument('--batch_size', type=int, default=120)
  parser.add_argument('--weight_decay', type=float, default=0)
  parser.add_argument('--pretrained_model', type=str, default=None)
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--solver_type', type=str, default='"SGD"')
  parser.add_argument('--delta', type=float, default=1e-8) #only for ADAM
  args = parser.parse_args()

  print("Feature process visual: %s" %args.feature_process_visual)
  print("Feature process language: %s" %args.feature_process_language)
  print("Loc: %s" %args.loc)
  print("Dropout visual %f" %args.dropout_visual)
  print("Dropout language %f" %args.dropout_language)
  print("Pretrained model %s" %args.pretrained_model)
  valid_loss_type = ['triplet', 'inter', 'intra']
  
  assert args.loss_type in valid_loss_type

  assert args.lw_inter >= 0
  assert args.lw_intra >= 0

  if args.loss_type == 'inter':
    args.lw_inter = 1
    args.lw_intra = 0 

  if args.loss_type == 'intra':
    args.lw_intra = 1
    args.lw_inter = 0 

  train_base = 'prototxts/train_clip_retrieval_%s.prototxt'
  solver_base = 'prototxts/solver_clip_retrieval_%s.prototxt' 
  deploy_base = 'prototxts/deploy_clip_retrieval_%s.prototxt' 
  snapshot_base = 'clip_retrieval_' 
  
  params = {}
  params['sentence_length'] = 50
  params['descriptions'] = args.train_json 
  params['features'] = args.train_h5 
  params['top_names'] = ['features_p', 'query', 'features_time_stamp_p', 'features_time_stamp_n']
  params['top_names_dict'] = {}
  for key in params['top_names']: params['top_names_dict'] = add_dict_values(key, params['top_names_dict']) 
  params['feature_process'] = args.feature_process_visual
  params['loc_feature'] = args.loc 
  params['language_feature'] = args.feature_process_language 
  params['loss_type'] = args.loss_type
  params['batch_size'] = args.batch_size  

  if args.loss_type in ['triplet', 'inter']:
    inter_top_name = 'features_inter'
    params['top_names'].append(inter_top_name)
    params['top_names_dict'] = add_dict_values(inter_top_name, params['top_names_dict'])
  if args.loss_type in ['triplet', 'intra']:
    intra_top_name = 'features_intra'
    params['top_names'].append(intra_top_name)
    params['top_names_dict'] = add_dict_values(intra_top_name, params['top_names_dict'])

  if args.language_layers in ['lstm', 'lstm_no_embed', 'gru', 'gru_no_embed']:
    params['top_names'].append('cont')
    params['top_names_dict'] = add_dict_values('cont', params['top_names_dict'])
    params['sentence_length'] = 50
    assert params['language_feature'] in ['recurrent_word', 'recurrent_embedding'] 

  top_size = len(params['top_names'])
 
  f = h5py.File(params['features'])
  feat = np.array(f.values()[0]) 
  f.close()
  visual_feature_dim = feature_process_dict[args.feature_process_visual](0,0,feat).shape[-1]

  language_processor = language_feature_process_dict[params['language_feature']](read_json(params['descriptions'])) 
  language_feature_dim = language_processor.get_vector_dim() 
  vocab_size = language_processor.get_vocab_size() 
  params['vocab_size'] = vocab_size
 
  pretrained_model_bool = False
  if args.pretrained_model:
    pretrained_model_bool = True 

  data_layer = 'dataLayer_ExtractPairedLanguageVision'
  tag = '%s%s%s_%s_lf%s_dv%s_dl%s_nlv%s_nll%s_edl%s_edv%s_pm%s_loss%s_lwInter%s' %(snapshot_base,args.tag,
                   args.feature_process_visual, args.feature_process_language, 
                   str(args.loc), str(args.dropout_visual), str(args.dropout_language), 
                   args.vision_layers, args.language_layers, 
                   '-'.join([str(a) for a in args.language_embedding_dim]), 
                   '-'.join([str(a) for a in args.visual_embedding_dim]), 
                   pretrained_model_bool, args.loss_type,
                   args.lw_inter)
  
  train_path = train_base %tag
  deploy_path = deploy_base %tag 
  solver_path = solver_base %tag 
  
  net = retrieval_net(args=args, data_layer=data_layer,param_str=params,params=params, top_size=top_size)
  net.visual_feature_dim = visual_feature_dim
  net.build_retrieval_model(params, train_path) 
  
  params['batch_size'] = 100

  net = retrieval_net(args=args, data_layer=data_layer,param_str=params,params=params, top_size=top_size, is_test=True)
  net.visual_feature_dim = visual_feature_dim
  net.batch_size=21
  net.build_retrieval_model_deploy(deploy_path, visual_feature_dim, language_feature_dim) 

  max_iter = args.max_iter 
  snapshot = args.snapshot
  stepsize = args.stepsize
  base_lr = args.base_lr 
  if os.path.exists("Cannot have the same solver path: %s" %solver_path):
    print("Cannot have the same solver path: %s" %solver_path)
  else:
    make_solver(solver_path, tag, train_path, [], **{'device_id': args.gpu, 'max_iter': max_iter, 'snapshot': snapshot, 'weight_decay': args.weight_decay, 'stepsize': stepsize, 'base_lr': base_lr, 'random_seed': args.random_seed, 'display': 10, 'type': args.solver_type, 'delta': args.delta, 'iter_size': 120/args.batch_size})
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu) 
    train_model(solver_path, args.pretrained_model)
