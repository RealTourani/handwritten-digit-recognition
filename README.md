

<h1  align="center">Handwritten digit recognition</h1>

<p  align="center"  width="100%">

</p>

  

## Table of contents

- [Explanation of the dataset](https://github.com/RealTourani/handwritten-digit-recognition/tree/main#explanation-of-the-dataset)

- [Labeling](https://github.com/RealTourani/handwritten-digit-recognition/tree/main#labeling)

- [Training and Model architacture](https://github.com/RealTourani/handwritten-digit-recognition#training-and-model-architacture)

- [Compiling the model](https://github.com/RealTourani/handwritten-digit-recognition#compiling-the-model)

- [Docker](https://github.com/RealTourani/handwritten-digit-recognition#docker)


  

## Overview <a name="Overview"></a>
in this project, we have worked on handwritten digits recognition using CNN & model sequential to train the model.

**Requirements:**

Before going deeper, you must install Python +3.7 & the following packages to run the code:

  

- CV2: `pip install OpenCV-python`

- Matplotlib: `pip install matplotlib`

- Tensorflow: `pip install TensorFlow`

- Sklearn: `pip install scikit-learn`

- Pillow: `pip install Pillow`

- path lib: `pip install pathlib`

  

## Explanation of the dataset<a name="Datasets"></a>

In this project, we have 4320 images that belong to 4 categories. each category represents numbers 1 to 4, and the shape of each image is 64x64.
First of all the original dataset needs to be rotated 180 degrees. So at this stage, we need to rotate all the images. To do this you can run [rotation.py](https://github.com/RealTourani/handwritten-digit-recognition/blob/main/rotation.py) or you can use the command line: `python rotation.py`. Make sure about the dataset path in the code.

  

## Labeling<a name="Labeling"></a>

To calculate and match each label for images just run [label.py](https://github.com/RealTourani/handwritten-digit-recognition/blob/main/label.py) or you are able to run the code using the command line: `python label.py` . make sure about the dataset path in the code.
This code will calculate the labels with the [calculate_label](def%20calculate_label%28img_basename%29:) function and then It will save the image path and its label in a CSV.([labels.csv](https://github.com/RealTourani/handwritten-digit-recognition/blob/main/labels.csv))
  
  

## Training and Model architacture<a name="training"></a>

For the training process, I have implemented 2 training files. [train.ipynb](https://github.com/RealTourani/handwritten-digit-recognition/blob/main/train.ipynb) can be used on Colab or Jupyter notebook. [train.py](https://github.com/RealTourani/handwritten-digit-recognition/blob/main/train.py) as you know is a single python file and you can use each one you prefer.
In this project, I have used model sequential because as I mentioned before the dataset is not complex to use pre-trained models. let's jump into the architecture!
At the **first layer**, we have 32 filters and the kernel size is (3x3). Also, the input shape is (64x64x1) and at the last the activation is relu.
Then we add Maxpooling to reduce the spatial dimensions of the output volume and the activation of Maxpooling is relu. 
The second layer is like the first one. So we add a Flatten layer to reduce the input data into a single dimension instead of 2 dimensions.
Also, we add a fully connected layer with 64 filters and relu activation.
Finally, we add a softmax function to convert a vector of values to a probability distribution.

### Compiling the model<a name="Training"></a>
 The optimizer for this project that is not complex is Adam.
 To fit the model I have set up the epoch to 15 and the batch size to 32.
 The accuracy on the test dataset is 0.98 and the loss is 0.085.
 Also, the F1 score and precision are 0.98.

## Detection<a name="detection"></a>
For detecting handwritten digits you can run [detect.py](https://github.com/RealTourani/handwritten-digit-recognition/blob/main/detect.py) in the command line like this:

    python detect.py -img test_image\1_1_A_0.jpg

## Docker<a name="Docker"></a>
If you want to use Docker, first of all, you have to clone the repository.
Then go to the project directory and build the docker image like this:

    docker build -t detection .
Then you can run the docker container like this:

    run --rm -v <path of test images folder on your local machine>:/app/images detection python /app/detect.py -img images/1_1_A_0.jpg

for example:

    run --rm -v C:/Users/HP/Desktop/test_image:/app/images detection python /app/detect.py -img images/1_1_A_0.jpg
