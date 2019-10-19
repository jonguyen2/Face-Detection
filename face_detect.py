#Jonathan Nguyen 
#CS 4432 
#Project 2 Face Detection

import cv2
import os
import numpy as np

#include XML files 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#create a path to the output file
output_file = 'test_case_output'

with open('all_file.txt') as file:
    test_cases_array = file.readlines()
    test_cases_array = [line.rstrip('\n') for line in open('all_file.txt')]

#create frame dimensions
frame = cv2.imread('test.jpg')
height, width, layers = frame.shape

video = cv2.VideoWriter('face_detection.avi', 0, 25, (width,height))

index = 0

#visit each file in all_text and create file name to be outputted after facial scan
for element in test_cases_array:
    input_file =  os.path.join('test_case_input', test_cases_array[index])
    img = cv2.imread(input_file)
    fileName = 'test_result' + str(index) + '.jpg'

    index = index + 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print ("Number of faces:", len(faces) ) 

    i = 0

#scan each image in all_file.txt for faces and eyes
    for (x,y,w,h) in faces:
	    cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
	    i = i + 1
	    cv2.putText(img, ('Face_%03d' % i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
	    roi_gray = gray[y:y + h, x:x + w]
	    roi_color = img[y:y + h, x:x + w]
	    eyes = eye_cascade.detectMultiScale(roi_gray)
	    for (ex,ey,ew,eh) in eyes:
	       cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

#write image to video, display image, write image to output file  'test_case_output'  
    video.write(img)
    cv2.imshow('img', img)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join(output_file , fileName), img)

cv2.destroyAllWindows()
video.release()

