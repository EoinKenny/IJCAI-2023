# Advancing-Post-Hoc-Case-Based-Explanation-with-Feature-Highlighting

![alt text](imgs/title2.png "Title")


This is the repository for the paper *"Advancing Post-Hoc Case-Based Explanation With Feature Highlighting"*[^1].

The paper was published in the main technical track at IJCAI 2023.

The paper proposes a way to augment typical example-based (i.e., case-based) explanation with feature highlighting. CNN specific and ANN agnostic approaches are proposed.

***

Reproduce

***

First download the ImageNet dataset [here]( https://www.kaggle.com/c/imagenet-object-localization-challenge), and the CUB-200 dataset [here](https://www.vision.caltech.edu/datasets/cub_200_2011/). You can put them where you like, but just make sure the dataloader in each functions.py file links to that directory. The default directories are the folders themselves (i.e., ImageNet and CUB).

Run
```
conda create -name env
conda activate env
pip -r install requirements.txt
```
Then go to the relevent folders and run all the scripts except the functions.py in each directory, and the results will reproduce.

### Bibtex

```
Coming Soon...
```


[^1]: Kenny, E.M., Delaney, E. and Keane, M. T., Advancing Post-Hoc Case-Based Explanation With Feature Highlighting. In * 32nd INTERNATIONAL JOINT CONFERENCE ON ARTIFICIAL INTELLIGENCE.* Macao, China, 2023.
