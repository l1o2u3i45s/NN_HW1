from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import os
import time
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("converted_keras/keras_Model.h5", compile=False)

# Load the labels
class_names = open("converted_keras/labels.txt", "r").readlines()

testDataDir = 'Data/testData'

testfiles = [f for f in os.listdir(testDataDir) if os.path.isfile(os.path.join(testDataDir, f))]

for filePath in testfiles:

    width = 224
    height = 224
    img = cv2.imread('Data/testData/'+ filePath)
    resizeImg = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
     
     # Make the image a numpy array and reshape it to the models input shape.
    resizeImgArr = np.asarray(resizeImg, dtype=np.float32).reshape(1, width, height, 3)

    # Normalize the image array
    resizeImgArr = (resizeImgArr / 127.5) - 1

    # Predicts the model
    prediction = model.predict(resizeImgArr)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    originType = filePath.split('_')[0] 
    resizeImg = cv2.putText(resizeImg,  class_name[2:].strip() + ':' + str(np.round(confidence_score * 100))[:-2] +'%', ( int(width*0.1), int(height * 0.8) ) , cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
    cv2.imshow("Image", resizeImg)
    
    time.sleep(1)
    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

cv2.destroyAllWindows()