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

def late_fusion(rgb_tag, flow_tag, split, iter, lambda_values):

    data = read_json('data/%s_data.json' %split)

    #read in raw scores from rgb/flow
    rgb_results = pkl.load(open('%s/%s_%s.p' %(result_dir, rgb_tag, split), 'rb'))
    flow_results = pkl.load(open('%s/%s_%s.p' %(result_dir, flow_tag, split), 'rb'))

    for l in lambda_values:
        all_segments = []
        print "Lambda %f: " %l
        for d in data:
            rgb_scores = rgb_results[iter][d['annotation_id']]
            flow_scores = flow_results[iter][d['annotation_id']]
            scores = l*rgb_scores + (1-l)*flow_scores 
            all_segments.append([possible_segments[i] for i in np.argsort(scores)])
        eval_predictions(all_segments, data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb_tag", type=str, default=None)
    parser.add_argument("--flow_tag", type=str, default=None)
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--iter", type=int, default=30000)
    parser.add_argument('--lambda_values', type=float, nargs='+', default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    args = parser.parse_args()

    late_fusion(args.rgb_tag, args.flow_tag, args.split, args.iter, args.lambda_values)
