import os
import numpy as np
import h5py
import pdb
import sys

feature_root = 'rgb_features/' #wherever your features are 
video_list = [feature_root + v for v in os.listdir(feature_root)]

def make_h5_dict(name):

   seconds_per_chunk = 5
   fps = 25.
   subsample = 5

   h5_file = h5py.File(video_list[0])
   features = np.array(h5_file['features'])
   h5_file.close()
   feature_dim = features.shape[1]

   feature_dict = {}
   for i, video in enumerate(video_list):

       sys.stdout.write('\r%d/%d' %(i, len(video_list)))
       f = video 
       average_frames = np.zeros((30/seconds_per_chunk, feature_dim))

       h5_file = h5py.File(f)
       features = np.array(h5_file['features'])
       h5_file.close()
       
       #extracted features at 25 fps and subsampled every 5 frames.  5 frames corresponds to one second.
       frames_per_chunk = int(seconds_per_chunk*(fps/subsample))
       count = 0
       for i in range(0, min(features.shape[0], frames_per_chunk*6), frames_per_chunk):
           average_frames[count, :] = np.mean(features[i:i+frames_per_chunk, :], axis = 0)
           count += 1

       video_name = video.split('fps25_')[-1].split('.h5')[0]
       feature_dict[video_name] = average_frames

   print "\n"
   f = h5py.File('data/%s.h5' %name, "w")
   for key in feature_dict.keys():
       dset = f.create_dataset(key, data=feature_dict[key])
   f.close()

make_h5_dict('average_rgb_feats')
