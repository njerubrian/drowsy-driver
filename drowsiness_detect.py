'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
calculations. Uses webcam video feed as input.'''

#Import necessary libraries
from geopy.geocoders import Nominatim
from pynotifier import Notification
from scipy.spatial import distance
from imutils import face_utils
import pandas as pd
import numpy as np
import getpass
import datetime
import geocoder
import pygame #For playing sound
import time
import dlib
import cv2
import os
import re



#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

#Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 0.3

#Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 10

#COunts no. of consecutuve frames below threshold value
COUNTER = 0

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

#Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#Start webcam video capture
video_capture = cv2.VideoCapture(0)

#video writing
#vid_fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
vid_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
outvid = cv2.VideoWriter('output_video.avi', vid_fourcc,20.0,(640,480), True)
#outvid = cv2.VideoWriter(args['ouput'], vid_fourcc,qargs['fps'],(w*2,h), True)


#Give some time for camera to initialize(not required)
time.sleep(1)

while(True):
    #Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #Detect facial points
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Detect if eye aspect ratio is less than threshold
        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:

                pygame.mixer.music.play(-1)
                cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)


                #write frame
                outvid.write(frame)

                #device datetime
                dev_date = datetime.date.today()
                dev_time = time.strftime('%H:%M:%S',time.localtime())
                dev_month = dev_date.strftime('%m')
                #device name
                device_name = os.environ['COMPUTERNAME']
                #location
                geoloc = geocoder.ip('me')
                lat_lng = geoloc.latlng
                locator = Nominatim(user_agent='myGeocoder')
                loc_convert = locator.reverse(lat_lng)
                device_raw = loc_convert.raw

                device_address = loc_convert.address

                #xy_split = lat_lng.split("''")
                length = len(lat_lng)
                middle_index = length//2
                x_lat1 = lat_lng[:middle_index]
                x_lat = str(x_lat1)
                y_long1 = lat_lng[middle_index:]
                y_long = str(y_long1)

                #current user
                dev_user = getpass.getuser()
                
                #driver state
                dev_state = 1

                to_DF = pd.DataFrame(columns=['Date', 'Time', 'Device', 'X_Lat', 'Y_Long', 'Location', 'State'])

                new_row = {'Date':dev_date, 'Time':dev_time, 'Device':device_name, 'X_Lat':x_lat, 'Y_Long':y_long, 'Location':device_address, 'State':dev_state}
                to_DF = to_DF.append(new_row, ignore_index=True)
                # to_Df = to_DF.append({'Date':'dev_date', 'Time':'dev_time', 'Device':'device_name', 'Location':'device_address', 'State':'dev_state'}, ignore_index=True)

                #to_DF = pd.DataFrame([[dev_date,dev_time,device_name,'xlat','ylon',device_address,dev_state],['date1b','time1b','c']],
                 #   index=['row1','row2'], columns=['Date', 'Time', 'Device', 'X_Lat', 'Y_Long','Location', 'State'])
                
                to_DF.to_excel(dev_user + '.xlsx', sheet_name=dev_month)


                Notification(title='DROWSY DRIVER', description='Kindly check on Driver {} driving {} .'.format(dev_user,device_name), duration = 20, urgency = Notification.URGENCY_CRITICAL).send()

                #if(to_DF.State == )


        else:
            pygame.mixer.music.stop()
            COUNTER = 0

    #Show video feed
    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()
