## ML Facial Expression Recognition ##
Categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral) using CNN.

#### Requirements: ####
- Numpy
- Pandas
- Matplotlib
- Keras (Theano as backend) 

## DATA SET USED
The data consists of 48x48 pixel greyscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
Fer.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. Fer.csv contains only the "pixels" column.
The training set consists of 28,709 examples. This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project.

Dataset Link : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## METHODOLOGY USED:
The given flowcharts explains the methodology shows two flowcharts one for saving the weights using hp5 file and another shows integration with live emotion detector.The given flowcharts describe how the data is manipulated to get the highest accuracy in data manipulation by reducing the validation loss and increasing the validation accuracy.

### BUILDING A CNN MODEL

![image](https://user-images.githubusercontent.com/23000971/33515363-23154b4c-d788-11e7-871f-91035b4ac64c.png)

### integrating with cam using opencv library

![image](https://user-images.githubusercontent.com/23000971/33515380-71ada25e-d788-11e7-9b6a-91fa90e85677.png)

## Plots

![image](https://user-images.githubusercontent.com/23000971/33515390-938a63a8-d788-11e7-89f6-3385137a131e.png)
![image](https://user-images.githubusercontent.com/23000971/33515392-9c9d5a68-d788-11e7-816c-010f0dded6e4.png)
