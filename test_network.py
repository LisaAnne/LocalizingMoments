import sys
sys.path.append('utils/')
from config import *
sys.path.append(pycaffe_dir)
import caffe
from utils import *
from data_processing import *
from eval import *
import numpy as np
import pickle as pkl
import copy
import argparse
caffe.set_mode_gpu()
caffe.set_device(device_id)

def test_model(deploy_net, snapshot_tag, 
               visual_feature='feature_process_norm', 
               language_feature='recurrent_embedding', 
               max_iter=30000, 
               snapshot_interval=30000, 
               loc=False,  
               test_h5='data/average_fc7.h5', 
               split='val'):

    params = {'feature_process': visual_feature, 'loc_feature': loc, 'loss_type': 'triplet', 
              'batch_size': 120, 'features': test_h5, 'oversample': False, 'sentence_length': 50,
              'query_key': 'query', 'cont_key': 'cont', 'feature_key_p': 'features_p',
              'feature_time_stamp_p': 'feature_time_stamp_p', 
              'feature_time_stamp_n': 'feature_time_stampe_n'}
    
    language_extractor_fcn = extractLanguageFeatures 
    visual_extractor_fcn = extractVisualFeatures 

    language_process = language_feature_process_dict[language_feature] 
    data_orig = read_json('data/%s_data.json' %split)
    language_processor = language_process(data_orig)
    data = language_processor.preprocess(data_orig)
    params['vocab_dict'] = language_processor.vocab_dict
    num_glove_centroids = language_processor.get_vector_dim()
    params['num_glove_centroids'] = num_glove_centroids
    thread_result = {}
  
    visual_feature_extractor = visual_extractor_fcn(data, params, thread_result)
    textual_feature_extractor = language_extractor_fcn(data, params, thread_result)
    possible_segments = visual_feature_extractor.possible_annotations
  
    snapshot = '%s/%s_iter_%%d.caffemodel' %(snapshot_dir, snapshot_tag)
 
    visual_feature_extractor = visual_extractor_fcn(data, params, thread_result)
    textual_feature_extractor = language_extractor_fcn(data, params, thread_result)
    possible_segments = visual_feature_extractor.possible_annotations
  
    all_scores = {}
    for iter in range(snapshot_interval, max_iter+1, snapshot_interval):
        sorted_segments_list = []
        net = caffe.Net(deploy_net, snapshot %iter, caffe.TEST)  
        all_scores[iter] = {}
   
        #determine score for segments in each video
        for id, d in enumerate(data):
  
            vis_features, loc_features = visual_feature_extractor.get_data_test({'video': d['video']})
            lang_features, cont = textual_feature_extractor.get_data_test(d)
        
            net.blobs['image_data'].data[...] = vis_features.copy()        
            net.blobs['loc_data'].data[...] = loc_features.copy()        
        
            for i in range(vis_features.shape[0]):
                net.blobs['text_data'].data[:,i,:] = lang_features
                net.blobs['cont_data'].data[:,i] = cont 
            
            top_name = 'rank_score'
            net.forward()
            sorted_segments = [possible_segments[i] for i in np.argsort(net.blobs[top_name].data.squeeze())]
            sorted_segments_list.append(sorted_segments)
            all_scores[iter][d['annotation_id']] = net.blobs[top_name].data.squeeze().copy()
    
            if id % 10 == 0:
                sys.stdout.write('\r%d/%d' %(id, len(data)))
    
        eval_predictions(sorted_segments_list, data)
            
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    pkl.dump(all_scores, open('%s/%s_%s.p' %(result_dir, snapshot_tag, split), 'w'))
    print "Dumped results to: %s/%s_%s.p" %(result_dir, snapshot_tag, split)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--deploy_net", type=str, default=None)
    parser.add_argument("--snapshot_tag", type=str, default=None)
    parser.add_argument("--visual_feature", type=str, default="feature_process_norm")
    parser.add_argument("--language_feature", type=str, default="recurrent_embedding")
    parser.add_argument("--max_iter", type=int, default=30000) 
    parser.add_argument("--snapshot_interval", type=int, default=30000) 
    parser.add_argument("--loc", dest='loc', action='store_true')
    parser.set_defaults(loc=False)
    parser.add_argument("--test_h5", type=str, default='data/average_fc7.h5')
    parser.add_argument("--split", type=str, default='val')

    args = parser.parse_args()

    test_model(args.deploy_net, args.snapshot_tag,
               visual_feature = args.visual_feature,
               language_feature = args.language_feature,
               max_iter = args.max_iter,
               snapshot_interval = args.snapshot_interval,
               loc = args.loc,
               test_h5 = args.test_h5,
               split = args.split) 
