# Dockerize Deep learning model

The objectibe of this repository is to give overview of deplyoing a pretrained deep learning computer vison model by dockerizing it, which is based on kornia framework which matches the given two image. 

[Kornia](https://kornia.readthedocs.io/en/latest/) is a computer vision framework built on top of pytorch 

The pretrained model which has been used is [kornia implementation](https://kornia.readthedocs.io/en/latest/applications/image_matching.html) of LoFTR: Detector-Free Local Feature Matching with Transformers. LoFTR can extract high-quality semi-dense matches even in indistinctive regions with low-textures, motion blur, or repetitive patterns. 



## To-Do
1. Install [Docker](https://docs.docker.com/engine/install/) as per your operating system.
1. Clone this repository by running ```git clone https://github.com/deshram/dockerize-image-matching-model```
2. Here I've used ```outdoor``` model weights, but you can also try [indoor](http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt) and [indoor new](http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt) weights which are trained on different dataset by downloading them in ```/models``` folder 
3. Build docker image by running ```docker build -t IMAGE_NAME .``` e.g. ```docker build -t image_mathcing .```
4. Run docker container to deploy on host ```docker run --gpus all -p HOST_PORT:CONTAINER_PORT IMAGE_NAME``` e.g. ```docker run --gpus all -p 5000:5000 image_matching```
PS: Use ```--gpus all``` if host machine contains GPU 
