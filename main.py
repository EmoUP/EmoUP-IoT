import os  
import cv2  
import time
import numpy as np  
from tensorflow.keras.models import model_from_json  
from tensorflow.keras.preprocessing import image  
import requests

#load model
model = model_from_json(open("./model.json", "r").read())
#load weights
model.load_weights('./model.h5')

face_haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')  

cap=cv2.VideoCapture(0)
previous = time.time()
delta = 0

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image  
    if not ret:
        continue

    current = time.time()
    delta += current - previous
    previous = current
    #print(delta)
    if delta > 600: #10 min delay
        delta = 0
        print("Predicting Emotion...")
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
            roi_gray=cv2.resize(roi_gray,(48,48))  
            img_pixels = image.img_to_array(roi_gray)  
            img_pixels = np.expand_dims(img_pixels, axis = 0)  
            img_pixels /= 255  
    
            predictions = model.predict(img_pixels)  
    
            #find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        print(predicted_emotion)
        url = "http://52.188.203.118:5000/users/update-emotion?user_id="+getserial()+"&emotion="+predicted_emotion+"&device=True"
        r = requests.post(url = url)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv0.imshow('Facial emotion analysis ',resized_img)
    
    if cv2.waitKey(10) == ord('q'): #wait until 'q' key is pressed  
        break  

cap.release()  
cv2.destroyAllWindows  

def getserial():
  # Extract serial from cpuinfo file
  cpuserial = "0000000000000000"
  try:
    f = open('/proc/cpuinfo','r')
    for line in f:
      if line[0:6]=='Serial':
        cpuserial = line[10:26]
    f.close()
  except:
    cpuserial = "ERROR000000000"
 
  return cpuserial