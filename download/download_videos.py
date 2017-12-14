''' 
Code to get the data in the DiDeMo video dataset.

Usage:

python download_videos.py  --download --video_directory DIRECTORY

will download videos from flickr to DIRECTORY

python download_videos.py  (without the download flag), will just check to see if a video is still on flickr.  
As of 08/20/2017, 316 of the original >10,000 videos have been removed from Flickr.
'''

import sys
import os
import json
import urllib
import urllib2
import argparse
import pdb
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--video_directory", type=str, default='videos/', help="Indicate where you want downloaded videos to be stored")
parser.add_argument("--download", dest="download", action="store_true")
parser.set_defaults(download=False)
args = parser.parse_args()
if args.download:
    assert os.path.exists(args.video_directory)

train_caps = read_json('data/train_data.json')
val_caps = read_json('data/val_data.json')
test_caps = read_json('data/test_data.json')

all_data = train_caps + val_caps + test_caps

link_dict = {}
for d in all_data:
    link_dict[d['dl_link']] = d['video']

count = 0
missing_videos_list = []
for dl_link, video_name in zip(link_dict.keys(), link_dict.values()):
    sys.stdout.write('\rDownlaoding video: %d/%d' %(count, len(link_dict.keys())))
    if args.download:
        try:
            response = urllib2.urlopen(dl_link)
            urllib.urlretrieve(response.geturl(), '%s/%s' %(args.video_directory, video_name))
        except:
            print "Could not download link: %s\n" %dl_link
    else:
        try:
            response = urllib2.urlopen(dl_link)
        except:
            print "Could not find link: %s\n" %dl_link
            missing_videos_list.append(video_name)
    count += 1
print "\n"
print "%d videos are missing" %len(missing_videos_list)


if len(missing_videos_list) > 0:
    write_txt = open('missing_videos.txt', 'w')
    for missing_video in missing_videos_list:
        write_txt.writelines('%s\n' %missing_video)
    write_txt.close()

