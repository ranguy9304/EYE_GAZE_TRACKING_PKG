#!/usr/bin/env python
from __future__ import print_function

from matplotlib import image
import rospy
import sys
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
# from numpy import argmax
# from beginner_tutorials.msg import Complex
from std_msgs.msg import String
from std_msgs.msg import Int32
import math
import functions_self as fun
font=cv2.FONT_HERSHEY_PLAIN


# init params #

pub,rate,cap,road,dif,rate_cap,rate_road,count,heat_init,ret_count,cords,detector,predictor=fun.init_var()

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (int(x),int(y)), (int(x_plus_w),int(y_plus_h)), color, 2)

    cv2.putText(img, label, (int(x-10),int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



while not rospy.is_shutdown():




    dist =lambda x1,y1,x2,y2 :(x1-x2)**2-(y1-y2)**2
    while True:

        r_,r_frame=road.read()
        _,frame=cap.read()

        #keep this batch in this order

        height,width,_=frame.shape
        r_frame = cv2.resize(r_frame,(int(width*0.5),int(height*0.5)),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        r_x,r_y,_=r_frame.shape
        # this batch in this order or else shit will stop

        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=detector(frame)

        if heat_init==0:
            heat= np.zeros((r_x+20, r_y+20))
            heat_init=1


        #---------------------- NOT LOOKING CODE -----------------------#
        if bool(faces):
            ret_count=ret_count+1
            pub.publish(" .. NORMAL .. ")


        

        if  ret_count!= count:
            pub.publish("404 ATTENTION NOT FOUND ")
            cv2.putText(frame,"SIR ROAD SAMNE HAI",(80,200),font,2,(0,0,255),3)
            ret_count=count
        ####################### NOT LOOKING DONE ###############################



        for face in faces:
            landmarks=predictor(gray,face)
        
            left_eye_region,nose_l,ear_l,lip_b,chin,left_point,right_point,center_top,center_bottom=fun.landmarks_init(landmarks=landmarks)
    

        #----------------- REGION AROUND EYE ----------------------#

            cv2.polylines(frame,[left_eye_region],True,(0,0,255),2)

            height,width,_=frame.shape

            r_height,r_width,_=r_frame.shape

            mask=np.zeros((height,width),np.uint8)
            cv2.polylines(mask,[left_eye_region],True,255,2)
            cv2.fillPoly(mask,[left_eye_region],255)

            left_eye=cv2.bitwise_and(gray,gray,mask=mask)


         ############### REGION AROUND EYE DONE #######################
            
         #---------------- EYE FRAMING ---------------------------#


            min_x=np.min(left_eye_region[:,0])
            max_x=np.max(left_eye_region[:,0])

            min_y=np.min(left_eye_region[:,1])
            max_y=np.max(left_eye_region[:,1])

            gray_eye=left_eye[min_y-5:max_y+5,min_x-5:max_x+5]


         #################### EYE FRAME DONE #######################

         #------------------ THRESHOLDING HERE -----------------------#


            gray_eye=cv2.GaussianBlur(gray_eye,(3,3),0)

            threshold_eye = cv2.adaptiveThreshold(gray_eye,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,103,-25)

            threshold_eye = cv2.flip(threshold_eye, 1)

            
         ######################## THRESHOLDING DONE #########################
         
            height,width=threshold_eye.shape

            eye=cv2.resize(gray_eye,None,fx=5,fy=5)

         #-------------------- DETECTING HOUGH CIRCLES ------------------#
            
            blur=cv2.GaussianBlur(threshold_eye,(3,3),0)
            circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,2,100,param1=100,param2=20,minRadius=17,maxRadius=27) 
         ###################### HOUGH CIRCLE DONE ##########################

         #------------------ DISTANCE OF THE CHEEK LINE --------------------#
            dist_cheek=fun.distance(nose_l[0],nose_l[1],ear_l[0],ear_l[1])
            # dist_cheek=(nose_l[0]-ear_l[0])**2+(nose_l[1]-ear_l[1])**2
            std_val_cheek=150000

            per_change=(float(dist_cheek-std_val_cheek)/std_val_cheek)*100

            if -60<=per_change <=-50:
                per_change=-50
            if per_change<-60:
                per_change=-55

         ################## DISTANCE OF THE CHEEK LINE ######################
         
         #--------------- PRINTING DETAILS ON SCREEN -----------------------#

            cv2.putText(frame,"ratios: "+str(float(dist_cheek)/150000),(200,200),font,2,(0,0,255),3)
            cv2.putText(frame,"difference of length: "+str(per_change),(50,400),font,2,(0,0,255),3)
            cv2.putText(frame,"length of the line: "+str(dist_cheek),(50,600),font,2,(0,0,255),3)


            cheek_l=cv2.line(frame,ear_l,nose_l,(0,255,0),2)

         #######################################################################

         #---------------------- DRAWING RATIO LINES (EYES) -------------------#
            no_div=10

            for i in range(no_div):
                div_w=float(width*i)/no_div
                cv2.line(blur, (int((div_w)+div_w*per_change/100),0), (int((div_w)+div_w*per_change/100),height), (255,255,255),5)

         ###########################################################################

         #----------------------------- RATIO LINES FOR (ROAD) --------------------#
            # for i in range(no_div):
            #     r_div_w=r_width*i/no_div
            #     cv2.line(r_frame, (int(r_div_w+r_div_w*per_change/100),0), (int(r_div_w+r_div_w*per_change/100),r_height), (255,255,255),5)
         ##############################################################################

            if circles is not None:
                circles=np.uint16(np.around(circles))


                for (x, y, r) in circles[0,:]:
                    shape_eye=blur.shape

                    #----------------- VERTI POSE -------------------------#

                    ratio,ratio_mult=fun.vert_pose(blur=blur,frame=frame,shape_eye=shape_eye,r_x=r_x,x=x,y=y)
                    #########################################################
                    

                    for i in range(no_div):

                        #----------------------CHECKING HORIZONTAL POSITION (GRID) ------------------------#

                        if float(width*i)/float(no_div)<=x<float(width*(i+1))/float(no_div):

                            y_cord_gaze,x_cord_gaze=fun.hor_pose(per_change=per_change,width=width,r_width=r_width,i=i,no_div=no_div,ratio=ratio,ratio_mult=ratio_mult,cords=cords,count=count)

                            #--------------------- SMA LIST ------------------#
                            cords.append((y_cord_gaze,x_cord_gaze))
                            if count>10:
                                cords.pop(0)

            #----------------------- FRAME AROUND THE GAZE ---------------------#

                            image=r_frame[int(x_cord_gaze)-208:int(x_cord_gaze)+208,int(y_cord_gaze)-208:int(y_cord_gaze)+208]

            ##############################################################################

            #------------------------------ SMA CIRCLES ------------------------------#
            sma_val=3
            if count<sma_val:

                print(".... reading values for sma ....")

            else:

                avg_y=0
                avg_x=0
                for i in range(sma_val):

                    avg_y=avg_y+cords[i][0]
                    avg_x=avg_x+cords[i][1]
                avg_y=avg_y/sma_val
                avg_x=avg_x/sma_val

                
                #sma circle
                    # cv2.circle(r_frame,[int(avg_y),int(avg_x)] , 50, [0,255,0], 5)



            # reference line #
            hor_line=cv2.line(frame,left_point,right_point,(0,255,0),2)
            ver_line=cv2.line(frame,center_bottom,center_top,(0,255,0),2)
            hor_line=cv2.line(frame,lip_b,chin,(0,255,0),2)
            

            landmarks=None

        fps = cap.get(cv2.CAP_PROP_FPS)
        r_fps = road.get(cv2.CAP_PROP_FPS)

#-------------------------------- OBJECT DETECTION ---------------------------#
    #BOTTLE NECK FOUND
        
        # try:
            
        #     Width = image.shape[1]
        #     Height = image.shape[0]
        #     scale = 0.00392

        #     classes = None

        #     with open('yolov3.names', 'r') as f:
        #         classes = [line.strip() for line in f.readlines()]

        #     COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        #     net = cv2.dnn.readNet('yolov3.weights.1', 'yolov3.cfg')

        #     blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=True)

        #     net.setInput(blob)

        #     outs = net.forward(get_output_layers(net))

        #     class_ids = []
        #     confidences = []
        #     boxes = []
        #     conf_threshold = 0.5
        #     nms_threshold = 0.4


        #     for out in outs:
        #         for detection in out:
        #             scores = detection[5:]
        #             class_id = np.argmax(scores)
        #             confidence = scores[class_id]
        #             if confidence > 0.5:
        #                 center_x = int(detection[0] * Width)
        #                 center_y = int(detection[1] * Height)
        #                 w = int(detection[2] * Width)
        #                 h = int(detection[3] * Height)
        #                 x = center_x - w / 2
        #                 y = center_y - h / 2
        #                 class_ids.append(class_id)
        #                 confidences.append(float(confidence))
        #                 boxes.append([x, y, w, h])


        #     indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        


        #     for i in indices:
        #         i=i[0]
        #         box = boxes[i]
        #         x = box[0]
                
        #         y = box[1]
        #         w = box[2]
        #         h = box[3]
        #         draw_prediction(image, class_ids[i], confidences[i], round(box[0]), round(box[1]), round(box[0]+box[2]), round(box[1]+box[3]))

        #     cv2.imshow("object detection", image)

        # except:
        #     print(".........error occured during object detection")

        cv2.circle(r_frame,(int(y_cord_gaze),int(x_cord_gaze)) , 50, (0,0,255), 5)

     ####################################### OBJECT DETECTION DONE ###############################################################3

        
        
        cv2.imshow("threshold",threshold_eye)
        cv2.imshow("blur",blur)
        cv2.imshow("Frame",frame)
        cv2.imshow("road",r_frame)

        count=count+1

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    rate.sleep()

    cap.release()
    road.release()
    cv2.destroyAllWindows()

    #####################################################################################################
    
 