from itertools import accumulate
import cv2 as cv
import os
from keras.models import load_model
import numpy as np
from numpy import size
from pygame import mixer
import time
import math
import PySimpleGUI as sg


sg.theme('DarkTanBlue')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text('Please select video source file or use camera')],
            [sg.Text('Choose a file: '), sg.Input(key="-IN-"), sg.FileBrowse(key="-IN-", file_types=[('Video Files', ['.avi', '.mkv', '.mp4', '.mpeg','.mpg', '.wav']),('AVI', '.avi'),('MKV', '.mkv'),('MP4', '.mp4'),('MPEG', ['.mpeg','.mpg']),('WAV', '.wav'), ('Any', '.*')])],
            [sg.Button('Use File'), sg.Button('Use Camera'), sg.Button('Cancel')],
            [sg.Text('Options: '), sg.Checkbox('Closed Eyes Model', default=True, key='useClosedEyeFeature'), sg.Checkbox('Mouth Yawn Model', default=True, key='useMouthFeature'), sg.Checkbox('Face Yawn Model', default=True, key='useFaceYawnFeature')],
            [sg.Checkbox('Alert Sounds', default=True, key='enableSound'), sg.Checkbox('Skip lagged video frames (Experimental)', default=False, key='skippingFrame')]]

# Create the Window
window = sg.Window('Drowsiness Detector', layout, size=(600,180))
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    
    video_file = None
    if event == 'Use File':
        if len(values["-IN-"])>0:
            video_file = values["-IN-"]
            print(values["-IN-"])
        else:
            sg.popup_ok('Please check video file path')
            continue
    elif event == 'Use Camera':
        video_file = None
        print("Camera")

    useClosedEyeFeature = values['useClosedEyeFeature']
    useMouthFeature = values['useMouthFeature']
    useFaceYawnFeature = values['useFaceYawnFeature']
    skippingFrame = values['skippingFrame']

    print("useClosedEyeFeature:"+str(useClosedEyeFeature)+", useMouthFeature:"+str(useMouthFeature)+", useFaceYawnFeature:"+str(useFaceYawnFeature))


    if values['enableSound']:
    #print(f"OpenCV version: {cv.__version__}")
        mixer.init()
        #mild_alert = mixer.Sound('ProjectCode\Alarm\Alarm_Short.mp3')
        mild_alert = mixer.Sound('ProjectCode\Alarm\Windows Background.wav')
        strong_alert = mixer.Sound('ProjectCode\Alarm\Bleep.mp3')

    # import video file
    #video_file = 'ProjectCode\Dataset\DSM_Dataset-HDTV720\Microsleep\Male\HDTV720\P1042751_720.mp4'
    #video_file = 'ProjectCode\Dataset\DSM_Dataset-HDTV720\Yawning\Male\HDTV720\P1042763_720.mp4'
    #video_file = 'ProjectCode\Dataset\DSM_Dataset-HDTV720\Yawning\Female\HDTV720\P1043065_720.mp4'

    face_cascade = cv.CascadeClassifier('ProjectCode\haar cascade files\haarcascade_frontalface_default.xml')
    left_eye_cascade = cv.CascadeClassifier('ProjectCode\haar cascade files\haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv.CascadeClassifier('ProjectCode\haar cascade files\haarcascade_righteye_2splits.xml')
    mouth_cascade = cv.CascadeClassifier('ProjectCode\haar cascade files\haarcascade_mouth.xml')


    eye_model = load_model('ProjectCode\models\Temp\ClosedEyeClassifier.h5')
    face_yawn_model = load_model('ProjectCode\models\Temp\FaceYawnClassifier.h5')
    mouth_yawn_model = load_model('ProjectCode\models\Temp\MouthYawnClassifier.h5')

    #eye_model = load_model('ProjectCode\models\Backup\cnnCat2.h5')
    path = os.getcwd()
    
    if video_file is None:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

    else:
        cap = cv.VideoCapture(video_file)

    font = cv.FONT_HERSHEY_TRIPLEX
    count=0
    alert_score=0
    alert_border=2
    accum_alert_score = 0
    r_eye_predict=[99]
    l_eye_predict=[99]
    eye_sampling_factor = 2
    mouth_sampling_factor = 2
    face_yawn_sampling_factor = 10
    margin = 1.2
    alert_threshold = 60
    accum_alert_threshold = 60

    eye_result = ''
    yawn_result = ''
    actual_frame = 0
    prev_frame_time = 0
    new_frame_time = 0

    start_time = time.time()

    while cap.isOpened() and cap is not None:
        try:

            ret, frame = cap.read()
            height,width = frame.shape[:2] 

            actual_frame += 1
            fps = cap.get(cv.CAP_PROP_FPS)
            new_frame_time = time.time()

            actual_fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            cv.putText(frame,'FPS: %.2f'%actual_fps, (width-80, 15), font, 0.25,(255,255,255),1,cv.LINE_AA)

            timed_frame = int(fps*(new_frame_time-start_time))
            
            if skippingFrame and video_file is not None and timed_frame > actual_frame:
                cap.set(cv.CAP_PROP_POS_FRAMES, timed_frame+2)
                prev_frame_time = time.time()
                print(f"Skipped to {timed_frame}")
                continue
            
            # initial
            isYawn_mouth = [0]
            isYawn_face = [0]
            left_eye = None
            right_eye = None
            ##################################################

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            try:
                faces = face_cascade.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.3, minSize=(25,25))
                
                cv.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv.FILLED )

                for (x,y,w,h) in faces:
                    margin_x = int(int(margin*w/4)/2)
                    margin_y = int(int(margin*h/4)/2)

                    cv.rectangle(frame, (x-margin_x,y-margin_y) , (x+w+margin_x,y+h+margin_y) , (100,100,100) , 1 )

                    face_gray = gray[y-margin_y:y+h+margin_y, x-margin_x:x+w+margin_x] #for grayscale image
                    face_color = frame[y-margin_y:y+h+margin_y, x-margin_x:x+w+margin_x] #for color image
                    

                    if useClosedEyeFeature:
                        left_eye = left_eye_cascade.detectMultiScale(face_gray, minNeighbors=5,scaleFactor=1.3)
                        right_eye =  right_eye_cascade.detectMultiScale(face_gray, minNeighbors=5,scaleFactor=1.3 )

                        for (rx,ry,rw,rh) in right_eye:
                            r_eye = face_gray[ry:ry+rh,rx:rx+rw]
                            r_eye_center_x = int((2*rx+rw)/2)
                            r_eye_center_y = int((2*ry+rh)/2)
                            radius = int((rw+rh)/4)
                            cv.circle(img=face_color, center= (r_eye_center_x, r_eye_center_y), radius=radius, color=(100,0,100), thickness=1)

                            count = count+1
                            
                            if int(time.time() * 1000000)%eye_sampling_factor == 0:
                                #r_eye = cv.cvtColor(r_eye,cv.COLOR_BGR2GRAY)
                                r_eye = cv.resize(r_eye,(24,24))
                                r_eye = r_eye/255
                                r_eye =  r_eye.reshape(24,24,-1)
                                r_eye = np.expand_dims(r_eye,axis=0)
                                r_eye_predict = np.argmax(eye_model.predict(r_eye),axis=1)
                            else:
                                r_eye_predict = r_eye_predict


                        for (lx,ly,lw,lh) in left_eye:
                            l_eye = face_gray[ly:ly+lh,lx:lx+lw]
                            l_eye_center_x = int((2*lx+lw)/2)
                            l_eye_center_y = int((2*ly+lh)/2)
                            radius = int((lw+lh)/4)
                            cv.circle(img=face_color, center= (l_eye_center_x, l_eye_center_y), radius=radius, color=(100,0,100), thickness=1)

                            count = count+1

                            if int(time.time() * 1000000)%eye_sampling_factor == 0:
                                #l_eye = cv.cvtColor(l_eye,cv.COLOR_BGR2GRAY)  
                                l_eye = cv.resize(l_eye,(24,24))
                                l_eye = l_eye/255
                                l_eye = l_eye.reshape(24,24,-1)
                                l_eye = np.expand_dims(l_eye,axis=0)
                                l_eye_predict = np.argmax(eye_model.predict(l_eye),axis=1)
                            else:
                                l_eye_predict = l_eye_predict

                    # Detecting Mouth
                    if useMouthFeature:
                        mouths = mouth_cascade.detectMultiScale(face_gray, scaleFactor=1.3, minNeighbors=38) #scaleFactor=1.4, minNeighbors=22


                        for (mx,my,mw,mh) in mouths:
                            mouth = face_gray[my:my+mh,mx:mx+mw]
                            count = count+1
                            if (int(time.time() * 1000000))%mouth_sampling_factor == 0:
                                #mouth = cv.cvtColor(mouth,cv.COLOR_BGR2GRAY)  
                                mouth = cv.resize(mouth,(48,48))
                                mouth = mouth/255
                                mouth = mouth.reshape(48,48,-1)
                                mouth = np.expand_dims(mouth,axis=0)
                                isYawn_mouth = np.argmax(mouth_yawn_model.predict(mouth),axis=1)
                            else:
                                isYawn_mouth = isYawn_mouth
                        
                            cv.rectangle(img=face_color, pt1=(mx,my), pt2=(mx+mw, my+mh), color=(0,0,100), thickness=1) #color = (B,G,R)

                    if useFaceYawnFeature:
                        if int(time.time() * 1000000)%face_yawn_sampling_factor == 0:
                            face = cv.resize(face_color[:,:,[2]],(48,48))
                            face = face/255
                            face = face.reshape(48,48,-1)
                            face = np.expand_dims(face,axis=0)
                            isYawn_face = np.argmax(face_yawn_model.predict(face),axis=1)
                        else:
                            isYawn_face = isYawn_face

                # Closed eye(s) check
                if(r_eye_predict[0]==0 and l_eye_predict[0]==0) or ((right_eye is not None and left_eye is None) and r_eye_predict[0]==0) or ((right_eye is None and left_eye is not None) and l_eye_predict[0]==0):
                    alert_score = alert_score+ int(4/(fps/30))
                    if int(time.time() * 1000000)%1 == 0:
                        eye_result = 'Eyes Closed'
                # if(r_eye_predict[0]==1 or l_eye_predict[0]==1):
                elif right_eye is not None and left_eye is not None:
                    alert_score=alert_score-int(2/(fps/30))
                    if int(time.time() * 1000000)%1 == 0:
                        eye_result = 'Eyes Open'
                
                # Yawn check
                if isYawn_mouth[0]:
                    if int(time.time() * 1000000)%1 == 0:
                        yawn_result = '*Yawning'
                    accum_alert_score=accum_alert_score+int(5/(fps/30))
                elif isYawn_face[0]:
                    accum_alert_score=accum_alert_score+int(5/(fps/30))
                    if int(time.time() * 1000000)%1 == 0:
                        yawn_result = '*Yawning'
                else:
                    if int(time.time() * 1000000)%1 == 0:
                        yawn_result = ''

                cv.putText(frame, eye_result,(10,height-20), font, 0.3,(255,255,255),1,cv.LINE_AA)
                cv.putText(frame, yawn_result,(100,height-20), font, 0.3,(255,255,255),1,cv.LINE_AA)

                alert_score = min(alert_score, 100)
                accum_alert_score = min(accum_alert_score, 100)

                if(alert_score<0):
                    alert_score=0
                if (accum_alert_score<0):
                    accum_alert_score = 0
                cv.putText(frame,'Drowsiness Level:'+str(int(alert_score)),(width-180, height-20), font, 0.3,(255,255,255),1,cv.LINE_AA)
                cv.putText(frame,'Fatigue Level:'+str(int(accum_alert_score)),(width-180, height-5), font, 0.3,(255,255,255),1,cv.LINE_AA)


                #Issuing Alerts
                if(alert_score>alert_threshold or accum_alert_score > accum_alert_threshold):
                    cv.imwrite(os.path.join(path,'image.jpg'),frame)

                    if(alert_border<10):
                        alert_border= min(alert_border+1, 10)
                    else:
                        alert_border=alert_border-1
                    
                        if(alert_border<2):
                            alert_border=2
                    
                    if alert_score > 95:
                        accum_alert_score=accum_alert_score + (0.2/(fps/30))
                        alert_color = (0,0,200)
                        try:
                            strong_alert.play()
                        except:
                            pass
                    
                    elif alert_score>alert_threshold:
                        accum_alert_score=accum_alert_score + (0.2/(fps/30))
                        alert_color = (30,30,180)
                        try:
                            mild_alert.play()
                            
                        except:
                            pass

                    if accum_alert_score > accum_alert_threshold/1.2:
                        alert_color = (200,0,200)
                        accum_alert_score = max(accum_alert_score-0.1, accum_alert_score/1.01)
                    cv.rectangle(frame,(0,0),(width,height),alert_color,alert_border)
                    

                else:
                    alert_border=2
                
                accum_alert_score = accum_alert_score- (1/pow(fps,1))

            except:
                pass

            cv.imshow('frame',frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        except AttributeError:
            print('Video ends')
            break
        except Exception as err:
            print(err)
            break
        except:
            break
    
    cap.release()
    cv.destroyAllWindows()

window.close()
