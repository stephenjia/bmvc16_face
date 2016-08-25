# bmvc16_face
# Introduction

This repository contains code to reproduce the face rotation experiments in section 4.1 of the paper [Towards Automatic Image Editing: Learning to See another You](http://homes.esat.kuleuven.be/~xjia/xjia_publications/xjia_bmvc16_facefinal.pdf), a BMVC 2016 paper by Amir Ghodrati\*, Xu Jia\*, Marco Pedersoli, Tinne Tuytelaars (\* Amir and Xu contributed equally).

We propose a method that aims at automatically editing an image by altering its attributes. More specifically, given an image of a certain class (e.g. a human face), the method should generate a new image as similar as possible to the given one, but with an altered visual attribute (e.g. the same face with a new pose or a different illumination).

If you use our code in your research, please cite following paper:
```
@inproceedings{face_bmvc16,
  author    = {Amir Ghodrati and Xu Jia and Marco Pedersoli and Tinne Tuytelaars},
  title     = {Towards Automatic Image Editing: Learning to See another You},
  booktitle = {BMVC},
  year      = {2016}
}
```
Example: <br />
![face rotation](https://lh3.googleusercontent.com/mM-K9csNYv_K52PP5g08ZCaoN3BwEVoXE6LxUgW_oZ4fMUlVtRlBc1hMKrv_G6riL17l1sXljORiR7Y=w1920-h1005) <br />
![face illumination] (https://lh3.googleusercontent.com/UhaP9aM_Ykyeia-7sL22IwqD-ntYlTTYCNyxT23cxdj2G9SuFT8vi0YKR9iRz3cMzBOAw5rCHcaa9VM=w1920-h1005-rw) <br />
![face inpainting](https://lh6.googleusercontent.com/fDsSIWvj2F6lA2v28xQlzKNKcQiOVUWH0SkDMFmunA3xk7Hi7oMD7hYMh52hFrSYrQLBOs-_iq4p-eI=w1920-h1005) <br />

# Installing
* Install [Lasagne](https://lasagne.readthedocs.io/en/latest/user/installation.html) and its prerequisites.
* Install cuda 7.5, Theano 0.9.0, cuDNN 5.0


# Demo
* Run the experiments for face rotation: <br />
First change the configuration of the experiment in ```config_stageX_color.py```
Then run
```
python train_stage1_color.py
```
to train the first stage model and write checkpoint files to the checkpoints directory. <br />
Then run
```
python train_stage2_color.py
```
to train the second stage model and write checkpoint files to the checkpoints directory. <br />
To evaluate the trained models on test data, run
```
python generate_triplet_demo_color.py
```


# Dataset
You need to first download MultiPIE dataset.
To crop and align faces, we use the code provided by Junho Yim for their cvpr15 paper titled ```Rotating Your Face Using Multi-task Deep Neural Network```. Please cite their paper if you use this code for face cropping and alignment.


# Results


