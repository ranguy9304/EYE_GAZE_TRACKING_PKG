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
font=cv2.FONT_HERSHEY_PLAIN


def init_var():
    rospy.init_node('topic_publisher')
    pub = rospy.Publisher('counter', String,queue_size=1)
    rate = rospy.Rate(2)

    cap=cv2.VideoCapture('v1_driver.mp4')
    road=cv2.VideoCapture('v1_road.mp4')

    dif =1500
    rate_cap=120 +dif
    rate_road=dif


    cap.set(cv2.CAP_PROP_POS_FRAMES, rate_cap)
    road.set(cv2.CAP_PROP_POS_FRAMES, rate_road)

    count = 0
    heat_init=0
    ret_count=0
    cords=[]

    detector =dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    return pub,rate,cap,road,dif,rate_cap,rate_road,count,heat_init,ret_count,cords,detector,predictor




def midpoint(p1,p2):
    return int((p1.x +p2.x)/2),int((p1.y+p2.y)/2)

def maxAndMin(featCoords,mult = 1):
    adj = 10/mult
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    maxminList = np.array([min(listX)-adj,min(listY)-adj,max(listX)+adj,max(listY)+adj])
    # print(maxminList)
    return (maxminList*mult).astype(int), (np.array([sum(listX)/len(listX)-maxminList[0], sum(listY)/len(listY)-maxminList[1]])*mult).astype(int)




   
def distance(x1,y1,x2,y2):
    return (x1-x2)**2+(y1-y2)**2


def vert_pose(blur,frame,shape_eye,r_x,x,y):
    y2=shape_eye[0]/2

    x2=shape_eye[1]

    x1=0
    y1=shape_eye[0]/2

    diff_in_x=x1-x2
    if diff_in_x ==0:
        diff_in_x=0.000001
    slope=float((y1-y2))/float(diff_in_x)
    deno=math.sqrt(1+math.pow(slope,2))

    if deno== 0:
        deno=0.0001



    dist_pup=float((y-y1)-(slope)*(x-x1))/float(deno)
    cv2.putText(frame,"dispalce: "+str(dist_pup),(50,700),font,2,(0,0,255),3)
    cv2.line(blur, (x1,int(y1)), (x2,int(y2)), (25,25,255),5)
    

    ratio_test=dist_pup/shape_eye[0]

    ratio=float(r_x*6)/10

    ratio_mult=ratio_test

    return ratio,ratio_mult



def hor_pose(per_change,width,r_width,i,no_div,ratio,ratio_mult,cords,count):
    
        rr_div_w=float(r_width*i)/float(no_div)
        rr_div_w_nex=float(r_width*(i+1))/float(no_div)
        y_cord_gaze=((rr_div_w+rr_div_w*per_change/100)+(rr_div_w_nex+rr_div_w_nex*per_change/100))/2
        x_cord_gaze=ratio+(ratio*ratio_mult)


        #--------------------- SMA LIST ------------------#
        cords.append((y_cord_gaze,x_cord_gaze))
        if count>10:
            cords.pop(0)

        return y_cord_gaze,x_cord_gaze


def landmarks_init(landmarks):
    left_eye_region=np.array([(landmarks.part(42).x , landmarks.part(42).y),
            (landmarks.part(43).x , landmarks.part(43).y),
            (landmarks.part(44).x , landmarks.part(44).y),
            (landmarks.part(45).x , landmarks.part(45).y),
            (landmarks.part(46).x , landmarks.part(46).y),
            (landmarks.part(47).x , landmarks.part(47).y),

            ],np.int32)

    nose_l=(landmarks.part(35).x,landmarks.part(35).y)
    ear_l=(landmarks.part(15).x,landmarks.part(15).y)


    lip_b=(landmarks.part(8).x,landmarks.part(58).y)

    chin=(landmarks.part(8).x,landmarks.part(8).y)

    left_point=(landmarks.part(42).x,landmarks.part(42).y)
    right_point=(landmarks.part(45).x,landmarks.part(45).y)


    center_top=midpoint(landmarks.part(43),landmarks.part(44))
    center_bottom=midpoint(landmarks.part(47),landmarks.part(46))


    return left_eye_region,nose_l,ear_l,lip_b,chin,left_point,right_point,center_top,center_bottom