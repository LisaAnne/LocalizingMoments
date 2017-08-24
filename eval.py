''' 
Code to evaluate your results on the DiDeMo dataset.
'''

from utils import *
import numpy as np

def iou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union
  
def rank(pred, gt):
      return pred.index(tuple(gt)) + 1
  
def eval_predictions(segments, data):
    '''
    Inputs:
	segments: For each item in the ground truth data, rank possible video segments given the description and video.  In DiDeMo, there are 21 posible moments extracted for each video so the list of video segments will be of length 21.  The first video segment should be the video segment that best corresponds to the text query.  There are 4180 sentence in the validation data, so when evaluating a model on the val dataset, segments should be a list of lenght 4180, and each item in segments should be a list of length 21. 
	data: ground truth data
    '''
    import pdb
    pdb.set_trace()
    average_ranks = []
    average_iou = []
    for s, d in zip(segments, data):
      pred = s[0]
      ious = [iou(pred, t) for t in d['times']]
      average_iou.append(np.mean(np.sort(ious)[-3:]))
      ranks = [rank(s, t) for t in d['times']]
      average_ranks.append(np.mean(np.sort(ranks)[:3]))
    rank1 = np.sum(np.array(average_ranks) <= 1)/float(len(average_ranks))
    rank5 = np.sum(np.array(average_ranks) <= 5)/float(len(average_ranks))
    miou = np.mean(average_iou)
  
    print "Average rank@1: %f" %rank1
    print "Average rank@5: %f" %rank5
    print "Average iou: %f" %miou
    return rank1, rank5, miou

if __name__ == '__main__':

    '''
    Example code to evaluate your model.  Below I compute the scores for the moment frequency prior
    You should see the following output when you run eval.py
        Average rank@1: 0.186842
        Average rank@5: 0.686842
    	Average iou: 0.252216
    '''

    train_data = read_json('data/train_data.json')
    val_data = read_json('data/val_data.json')
    moment_frequency_dict = {}
    for d in train_data:
        times = [t for t in d['times']]
        for time in times:
            time = tuple(time)
            if time not in moment_frequency_dict.keys():
                moment_frequency_dict[time] = 0
            moment_frequency_dict[time] += 1
    
    prior = sorted(moment_frequency_dict, key=moment_frequency_dict.get, reverse=True)
    prediction = [prior for d in val_data]

    eval_predictions(prediction, val_data)


