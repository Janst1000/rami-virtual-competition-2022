# rami-virtual-competition-2022

## Setup of Conda Environment TODO

```
conda create -n rami --file env.yml
conda activate rami
pip install -r requirements.txt
```

## How does it work?

We are using two seperately trained yolo-v7 models. One is trained on preprocessed images to detect numbers and the other one is trained on unprocessed images to detect the balls and their color underwater. We seperated them as it was very hard to detect numbers without preproccesing and it is hard to detect a balls color after preprocessing as our preprocessed image is not a color image.

To preprocess the images we do the following steps:

1. remove green tint with [this function](https://stackoverflow.com/questions/64762020/how-to-desaturate-one-color-in-an-image-using-opencv2-in-python)
2. remove color cast with [this function](https://stackoverflow.com/questions/70876252/how-to-do-color-cast-removal-or-color-coverage-in-python)
3. convert to hsv and increase contrast in s channel
4. lab decomposition and only taking max between l and b
5. normalize our max_lb image
6. convert back to bgr for yolo

After this we first look if we can find any number as otherwise any gradients might be recognized as balls.
If we don't find any numbers with a certainty of at least 75% we look for balls using the original input image

Finally we return our prediction in the format ``<class>_<subclass>``

Here are some examples of that

```
1_red	//balls are class 1 and the subclass was red
2_2 	//numbers are class 2  and the subclass was the number "2"
```

We also return the xy coordinated in the frame and the certainty of our prediction

We are also using a modiefied version of the yolov7 code as the official version does not allow real time predictions. We modified it in Detection.py and also removed a lot of the things that we did not need from the original detection script.

## Results

When creating a test set with ``create_testset.ipynb`` and then running with the created testset we usually get an accuracy of about 98%. Unfortunatly we cannot get any accuracy results on the location of our predictions but when looking at the output images they look very good.

Here is an example of our code running and predicting:

![1676908127685](image/README/1676908127685.png)
