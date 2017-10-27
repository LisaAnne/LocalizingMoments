# Localizing Moments in Video with Natural Language.

Hendricks, Lisa Anne, et al. "Localizing Moments in Video with Natural Language." ICCV (2017).

Find the paper [here](https://arxiv.org/pdf/1708.01641.pdf) and the project page [here.](https://people.eecs.berkeley.edu/~lisa_anne/didemo.html)

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

I will release code to replicate my models shortly.  For now please see "eval.py" for instructions on how to evaluate models.

You can access my deploy prototxt and models [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/models/).  I have deploy prototxts/models for both RGB and flow inputs.  
The inputs to the models are: 

* image_data: pre-extracted features from either rgb or flow (see below for links to pre-extracted features)
* loc_data: start and end points for video moments (normalized by video length)
* text_data: glove embeddings for words in a sentence.  If a word is not in glove, it is represented as a zero vector.  The input is TxNx300 where T is the max number of words in a query (I use 50) and N is the batch size.  300 is the size of the Glove embedding.  If a sentene is of length t < T, then the last t entries include Glove embeddings for words in the sentences.
* cont_data: TxN matrix which indicates when sentences begin where T is max sentence length and N is batch size.  If sentence n is of length t, cont_data[:-(t-1), n] = 1 and cont_data[:(t+1), n] = 0.  Caffe can only handle fixed length sequences, so the cont_data input provides a way to flush the LSTM cell when a sentence begins.

## Dataset

### Annotations

To access the dataset, please look at the json files in the "data" folder.  Our annotations include descriptions which are temporally grounded in videos.  For easier annotation, each video is split into 5-second temporal chunks.  The first temporal chunk correpsonds to seconds 0-5 in the video, the second temporal chunk correpsonds to seconds 5-10, etc.  The following describes the different fields in the json files:

* annotation_id: Annotation ID for description
* description: Description for a specific video segment
* video: Video name
* times: Ground truth time points marked by annotators.  The time points indicate which chunk includes the start of the moment and which chunk includes the end of the moment.  An annotation of (3,3) indicates that a moment starts at 3x5=15 seconds and ends at (3+1)x5=20 seconds.  An annotation of (1,4) indicates that a moment starts at 1x5=5 seconds and ends at (4+1)x5=20 seconds.  Note that the last segment is not always 5 seconds long.  For example, for a video which is 28.2 seconds long, the annotation (5,5) will correpsond to 5x5=25 seconds to min((5+1)x5 seconds, video length) = 28.2 seconds.  Some videos are longer than 30 seconds.  These videos were truncated to 30 seconds during annotation.
* download_link: A download link for the video
* num_segments:  Some videos are a little shorter than 25 seconds, so were split into five temporal chunks instead of six.

### Getting the Videos

Use the script download_videos.py:
`python download_videos.py  --download --video_directory DIRECTORY`

There are some videos which have been removed from Flickr (~3% of the original videos).  You may access them [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/missing_videos/missing_videos.zip).

Please contact me if you find more videos are missing.  I originally downloaded the videos provided in the YFCC100M dataset (2nd entry for lines in "video_licenses.txt"), but many of these links no longer work.  The script I used to download the videos should work, but let me know if you run into any issues.  If you have a better understanding of the FlickrAPI (or are familiar with YFCC100M) and know why I can no longer download videos using the original download links, let me know.

You can view the Creative Commons licenses in "video_licenses.txt".

### Pre-Extracted Features

You can access preextracted features for RGB [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_fc7.h5) and for flow [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_global_flow.h5).
