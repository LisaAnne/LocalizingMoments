# Distinct Describable Moments (DiDeMo) dataset

## License

The annotations in this repository are covered by the [Creative Commons Attribution-NonCommercial-ShareAlike license (CC BY-NC-SA)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## Annotations

To access the dataset, please look at the json files in the "data" folder.  Our annotations include descriptions which are temporally grounded in videos.  For easier annotation, each video is split into 5-second temporal chunks.  The first temporal chunk correpsonds to seconds 0-5 in the video, the second temporal chunk correpsonds to seconds 5-10, etc.  The following describes the different fields in the json files:

* annotation_id: Annotation ID for description
* description: Description for a specific video segment
* video: Video name
* times: Ground truth time points marked by annotators.  The time points indicate which chunk includes the start of the moment and which chunk includes the end of the moment.  An annotation of (3,3) indicates that a moment starts at 3x5=15 seconds and ends at (3+1)x5=20 seconds.  An annotation of (1,4) indicates that a moment starts at 1x5=5 seconds and ends at (4+1)x5=20 seconds.  Note that the last segment is not always 5 seconds long.  For example, for a video which is 28.2 seconds long, the annotation (5,5) will correpsond to 5x5=25 seconds to min((5+1)x5 seconds, video length) = 28.2 seconds.  Some videos are longer than 30 seconds.  These videos were truncated to 30 seconds during annotation.
* download_link: A download link for the video.  Unfortunately, this download link does not work for many Flickr videos anymore.  See "Getting the Videos" for more details.
* num_segments:  Some videos are a little shorter than 25 seconds, so were split into five temporal chunks instead of six.

## Getting the Videos

1.  Download videos from AWS (preferred method).  YFCC100M images and videos are stored on AWS [here](https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/).  Because many videos have been deleted off of Flickr since I collected the dataset, it is best to access the videos stored on AWS instead of trying to download directly from Flickr.  To download the videos used in my dataset use the script download_videos_AWS.py:

`python download_videos_AWS.py --download --video_directory DIRECTORY`

There are 13 videos which are not on AWS which you may download from my website [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/missing_videos/missing_videos_AWS.zip) (I don't have enough space to store all the videos on my website -- Sorry!)

2.  Download videos directly from Flickr.  This is what I did when collecting the dataset, but now many Flickr videos have been deleted and many people have had issues running my download script.  To download videos directly from Flickr:

Use the script download_videos.py:
`python download_videos.py  --download --video_directory DIRECTORY`

When I originally released the dataset, ~3% of the original videos had been deleted from Flickr.  You may access them [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/missing_videos/missing_videos.zip).  If you find that more videos are missing, please download the videos via the AWS links above.

3.  Download from [Google Drive](https://drive.google.com/drive/u/1/folders/1_oyJ5rQiZboipbMl6tkhY8v0s9zDkvJc).  

You can view the Creative Commons licenses in "video_licenses.txt".

