import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)



#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
#Now we need to scale the data to make sure that the data points in X and Y are equal so we'll divide them
#using 255 which is the maximum pixel of the image
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
#After scaling the data we have to fit it inside our model so that it can give output with maximum accuracy
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

#Now, we have our classifier ready. Using this classifier, if we have an image, 
#can we predict the digit mentioned on the image?


#If we remember what we did earlier, we were using CV2 and using CV2, we were using our device's camera and
# capturing each frame. Now, each frame was an image where we were doing some processing and then predicting
# the value from it.Let's create a function to do that.
# We'll call it get _prediction which will take the image as the parameter and make a prediction. This function will
# take the image and convert it into a scalar quantity and then make it grey so that the colors don't affect the
# prediction. And then resize it into 28 by 28 scales. Then using the percentile function get the minimum pixel
# and then using the clip function give each image a number. And then get the maximum pixel and make an array of
#this. And then create a test sample of it and make predictions based on that sample. And then finally return the
# test prediction.

def get_prediction(image):
    #provide image
    im_pil = Image.open(image)
    #convert into gray scale and scaler quantity
    image_bw = im_pil.convert('L')
    #resize all image in 28X28 form
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    #get the min pixel
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    #give each image a number
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    #maximum pixel
    max_pixel = np.max(image_bw_resized)
    
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]