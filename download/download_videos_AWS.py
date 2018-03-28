''' 
Code to get the data in the DiDeMo video dataset using the videos stored on AWS.

Usage:

python download_videos_AWS.py  --download --video_directory DIRECTORY

will download videos from flickr to DIRECTORY

python download_videos_AWS.py  
'''
import sys
sys.path.append('.')
from utils.utils import *
import urllib
import urllib2
import pdb
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--video_directory", type=str, default='videos/', help="Indicate where you want downloaded videos to be stored")
parser.add_argument("--download", dest="download", action="store_true")
parser.set_defaults(download=False)
args = parser.parse_args()
if args.download:
    assert os.path.exists(args.video_directory)

multimedia_template = 'https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/%s/%s/%s.mp4'

splits = ['test', 'val', 'train']
data_template = 'data/%s_data.json'
caps = [] 
for split in splits:
     caps.extend(read_json(data_template %split))
videos = set([cap['video'] for cap in caps])
 
def read_hash(hash_file):
    lines = open(hash_file).readlines()
    yfcc100m_hash = {}
    for line_count, line in enumerate(lines):
         sys.stdout.write('\r%d/%d' %(line_count, len(lines)))
         line = line.strip().split('\t')
         yfcc100m_hash[line[0]] = line[1]
    print "\n"
    return yfcc100m_hash

def get_aws_link(h):
     return multimedia_template %(h[:3], h[3:6], h) 

yfcc100m_hash = read_hash('data/yfcc100m_hash.txt')

missing_videos = []

for video_count, video in enumerate(videos):
     sys.stdout.write('\rDownloading video: %d/%d' %(video_count, len(videos)))
     video_id = video.split('_')[1]
     link = get_aws_link(yfcc100m_hash[video_id])
     import pdb; pdb.set_trace()
     if args.download:
        try:
            response = urllib2.urlopen(link)
            urllib.urlretrieve(response.geturl(), '%s/%s.mp4' %(args.video_directory, video))
        except:
            print "Could not download link: %s\n" %link
     else:
         try:
             response = urllib2.urlopen(link)
         except:
             missing_videos.append(video)
             print "Could not find link: %s\n" %link

if len(missing_videos) > 0:
    write_txt = open('missing_videos.txt', 'w')
    for video in missing_videos:
        write_txt.writelines('%s\n' %video)
    write_txt.close()
