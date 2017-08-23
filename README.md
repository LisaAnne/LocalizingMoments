# Localizing Moments in Video with Natural Language.

Hendricks, Lisa Anne, et al. "Localizing Moments in Video with Natural Language." ICCV (2017).

[Find the paper here.](https://arxiv.org/pdf/1708.01641.pdf) and the project page [here.](https://people.eecs.berkeley.edu/~lisa_anne/didemo.html)

```
@inproceedings{hendricks17iccv, 
        title = {Localizing Moments in Video with Natural Language.}, 
        author = {Hendricks, Lisa Anne and Wang, Oliver and Shechtman, Eli and Sivic, Josef and Darrell, Trevor and Russell, Bryan}, 
       booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)}, 
       year = {2017} 
}
```

License: BSD 2-Clause license

## Running the Code

I will release code to replicate my models shortly.  For now please see "eval.py" 

## Dataset

### Annotations

To access the dataset, please look at the json files in the "data" folder.  Our annotations include descriptions which are temporally grounded in videos.  For easier annotation, each video is split into 5-second temporal chunks.  The first temporal chunk correpsonds to seconds 0-5 in the video, the second temporal chunk correpsonds to seconds 5-10, etc.  The following describes the different fields in the json files:

* annotation_id: Annotation ID for description
* description: Description for a specific video segment
* video: Video ID
* times: Ground truth time points marked by annotators.  The time points indicate which chunk includes the start of the moment and which chunk includes the end of the moment.  An annotation of (3,3) indicates that a moment starts at second 3x5=15 seconds and ends at second (3+1)x5=20 seconds.  An annotation of (1,4) indicates that a moment starts at second 1x5=5 seconds and ends at second (4+1)x5=20 seconds.
* download_link: A download link for the video
* num_segments:  Some videos are a little shorter than 25 seconds, so were split into five GIFs instead of six.

### Getting the Videos

Use the script download_videos.py:
`python download_videos.py  --download --video_directory DIRECTORY`

There are some videos which have been removed from Flickr (~3% of the original videos).  You may access them at my website: https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/missing_videos/missing_videos.zip

Please contact me if you find more videos are missing.  I originally downloaded the videos provided in the YFCC100M dataset (10th entry for lines in "video_licenses.txt"), but many of these links no longer work.  The script I used to download the videos should work, but let me know if you run into any issues.  If you have a better understanding of the FlickrAPI (or are familiar with YFCC100M) and know why I can no longer download videos using the original download links, let me know.

You can view the Creative Commons licenses and all metadata in "video_licenses.txt".

