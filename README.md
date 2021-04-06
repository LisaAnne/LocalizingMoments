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

**Preliminaries**:  I trained all my models with the [BVLC caffe version](https://github.com/BVLC/caffe).  Before you start, look at "utils/config.py" and change any paths as needed (e.g., perhaps you want to point to a Caffe build in a different folder).

**Evaluation**

Look at "utils/eval.py" if you would like to evaluate a model that you have trained.  Below are instructions to eval the models I proposed in my paper:

* ~Download data/models with "download/get_models.sh".  This should download models I trained and pre-extracted features.  Note that I retrained my models before releasing and the numbers are slightly different than those reported in the paper.~
* My website got deleted when I graduated.  Please find data on a google drive [here](https://drive.google.com/drive/u/1/folders/1heYHAOJX0mdeLH95jxdfxry6RC_KMVyZ).
* Run "test_network.sh".  This will run both RGB and flow models on the val and test sets.  It will also produce the scorse for the fusion model.  

You should get the following outputs:

| | Rank@1 | Rank@5 | mIOU |
| --- | --- | --- | --- |
| RGB val | 0.2442 | 0.7540 | 0.3739 |
| Flow val | 0.2626 | 0.7839 | 0.4015 |
| Fusion val (lambda 0.5) | 0.2765 | 0.7961 | 0.4191 |
| RGB test | 0.2312 | 0.7336 | 0.3549 |
| Flow test | 0.2583 | 0.7540 | 0.3894 |
| Fusion test (lambda 0.5) | 0.2708 | 0.7853 | 0.4053 |

**Training**

Use "run_job_rgb.sh" to train an RGB model and "run_job_flow.sh" to train a flow model.  You should be able to rerun these scripts and get simiar numbers to those reported in the paper.

## Dataset

You can find details on accessing the Distinct Describable Moments (DiDeMo) dataset are [here](data/README.md).

### Pre-Extracted Features

You can access preextracted features for RGB [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_fc7.h5) and for flow [here](https://people.eecs.berkeley.edu/~lisa_anne/didemo/data/average_global_flow.h5).  These are automatically downloaded in "download/get_models.sh".  To extract flow, I used the code [here](https://github.com/wanglimin/dense_flow).

I provide re-extracted features in the Google Drive above.  You can use [this script](https://github.com/LisaAnne/LocalizingMoments/blob/master/make_average_video_dict.py) to create a dict with averaged RGB features and [this script](https://github.com/LisaAnne/LocalizingMoments/blob/master/make_average_video_dict_flow.py).  The average features will be a bit different than the original release, but did not influence any trends in the results.
