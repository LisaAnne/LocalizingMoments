# LocalizingMoments

Hendricks, Lisa Anne, et al. "Localizing Moments in Video with Natural Language." ICCV (2017).

[Find the paper here.](https://arxiv.org/pdf/1708.01641.pdf)

```
@inproceedings{hendricks16cvpr, 
        title = {Localizing Moments in Video with Natural Language.}, 
        author = {Hendricks, Lisa Anne and Wang, Oliver and Shechtman, Eli and Sivic, Josef and Darrell, Trevor and Russell, Bryan}, 
       booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)}, 
       year = {2017} 
}
```

## Running the Code

To start, please run "./setup.sh". 

## Dataset

### Annotations

To access the dataset, please look at the json files in the "data" folder.  The json files contain a list of data points which are stored as dict structures with the following fields:
	annotation_id: Annotation ID for data point
        description: Description for a specific video segment
 	video: Video id
	times: Ground truth time points marked by annotators
        download_link: a download link for the video
        num_segments:  how many GIFs were shown to annotators.  Some videos are a little shorter than 25 seconds, so were split into five GIFs instead of six.

### Getting the Videos

Use the script download_videos.py:
python download_videos.py  --download --video_directory DIRECTORY

There are some videos which have been removed from Flickr (~3% of the original videos).  You may access them at my website: https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/missing_videos/missing_videos.zip

Please contact me if you find more videos are missing.  I originally downloaded the videos provided in the YFCC100M dataset (10th entry for lines in "video_licenses.txt"), but many of these links no longer work.  The srcript I used to download the videos should work, but let me know if you run into any issues.  If you have a better understanding of the FlickrAPI (or are familiar with YFCC100M) and know why I can no longer download videos using the original download links, let me know.

You can view the Creative Commons licenses and all metadata in "video_licenses.txt".

