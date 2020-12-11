from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np


def skin_detect(image):
	
	img = image
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	(H,S,V) = cv2.split(hsv)
	ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	(Y,Cr,Cb) = cv2.split(ycrcb)
	skin = np.zeros(H.shape,dtype = np.uint8)
	
	skin = 1*(H>1)*(H<23)*(Cr>133)*(Cr<177)*(Cb>94)*(Cb<125)
	skin = skin*255
	skin = skin.astype(np.uint8)
	
	kernel = np.ones((3,3),np.uint8)
	erosion = cv2.erode(skin,kernel)
	dilation = cv2.dilate(erosion,kernel)
	skin = dilation
	
	mask = cv2.bitwise_and(img,img,mask=skin)
	return skin,mask

def skin_threshold(image):
	ycrcb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
	(Y,Cr,Cb) = cv2.split(ycrcb)
	Cr1 = cv2.GaussianBlur(Cr,(5,5),0)
	_, skin = cv2.threshold(Cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	kernel = np.ones((3,3),np.uint8)
	erosion = cv2.erode(skin,kernel)
	dilation = cv2.dilate(erosion,kernel)
	skin = dilation
	
	mask = cv2.bitwise_and(image,image,mask=skin)
	return skin, mask
	
def skin_ellipse(image):
	skinCrCbHist = np.zeros((256,256),dtype = np.uint8)
	cv2.ellipse(skinCrCbHist,(113,155),(23,25),43,0,360,(255,255,255),-1)
	ycrcb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
	(Y,Cr,Cb) = cv2.split(ycrcb)
	skin = 255*(1*skinCrCbHist[Cr,Cb]>0)
	skin = skin.astype(np.uint8)
	
	kernel = np.ones((3,3),np.uint8)
	erosion = cv2.erode(skin,kernel)
	
	dilation = cv2.dilate(erosion,kernel)
	skin = dilation
	
	mask = cv2.bitwise_and(image,image,mask=skin)
	return skin, mask
	
			
camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = 32
camera.hflip = False
camera.vflip = True
camera.rotation = 90
rawCapture = PiRGBArray(camera,size=(320,240))

time.sleep(0.1)
num = 0
for frame in camera.capture_continuous(rawCapture,format = "bgr", use_video_port = True):
	image = frame.array
	image = cv2.bilateralFilter(image,9,75,75) #Filter
	#image = cv2.GaussianBlur(image,(3,3),0)
	
	image2 = image.copy()
	
	cv2.imshow("Original",image)
	skin, mask= skin_threshold(image)
	#skin, mask= skin_detect(image)
	#cv2.imshow("Frame_1",skin)
	cv2.imshow("MASK",mask)
	
	h = cv2.findContours(skin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	contours = h[0] 
	ret = np.ones(image.shape,np.uint8)
	cv2.drawContours(ret,contours,-1,(255,0,0),1)
	cv2.imshow("Contour",ret)
	
	
	areas = list()
	for i,cnt in enumerate(contours):
		areas.append(cv2.contourArea(cnt))
	ll = areas.index(max(areas))	
	
	cnt = contours[ll]
	hull = cv2.convexHull(cnt,returnPoints = False)
	defects = cv2.convexityDefects(cnt,hull)
	
	if defects is not None:
		for i in range(defects.shape[0]):
			s,e,f,d = defects[i,0]
			start = tuple(cnt[s,0]) 
			end = tuple(cnt[e,0])
			far = tuple(cnt[f,0])
			cv2.line(image,start,end,(255,255,0),2)
			cv2.circle(image,far,5,(0,0,255),-1)
	cv2.imshow("Final",image)
	key = cv2.waitKey(1) & 0xFF
	
	if key==ord('s'):
		strs = '/home/pi/New/number4/' +str(num)+'new'+ '.jpg'
		cv2.imwrite(strs,image2)
		
		num+=1
	rawCapture.truncate(0)
	
	if key==ord("q"):
		break
	
