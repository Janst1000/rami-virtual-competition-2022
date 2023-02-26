# rami-virtual-competition-2022

This repository is using yolo-v7 models and image transformations to predict buoys and numbers on pipes in marine robotics datasets. The dataset that we used was from the rami virtual competition in 2022. Unfortunatly we are not able to predict all classes that were featured in the dataset yet but we can predict the numbers and buoys from class 1 and class 2. some of the other classes could be implemented this way in the future however class 3 images cannot be implemented this way as we do not have enough training data.

## Setup of Conda Environment

```
conda create -n rami --file env.yml
conda activate rami
pip install -r requirements.txt
```

## Usage

Since this project was based on the rami-virtual-competition in 2022 we used their dataset which all came as seperate images to detect object on. However this code could be modified in the future to take the input stream from a camera or ros camera topic.

To run the script with a custom dataset, you will have to adjust a few things. We need a ``ground_truth.txt`` file in the same directory that has our input images. In this file we put the expected class on each row. To see how it is supposed to look like you can find an example one in ``rami_marine_dataset/test/ground_truth.txt``

To run the script run the following:

```
python3 test.py -i <input-dir> -gt
```

The following flags can be given to the script

```
-i	input directory (required)
-o	should and output video be created
-gt	a ground_truth.txt is in the input dir and we will calculate the accuracy of our predictions
--show	shows the resulting image with prediction
```

So for a usual run of our example that produced the output below we would run:

```
python3 test.py -i rami_marine_dataset/test -gt --show
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
