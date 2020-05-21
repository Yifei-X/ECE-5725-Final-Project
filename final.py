from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

def skin_detect(image):#method 1 of skin detection using normal threshold on HSV and YCrCb
	
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

def skin_threshold(image): # method 2 using OTSU Threshold on Cr(YCrCb)
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
	
def skin_ellipse(image):# method 3 of using eliipse method on YCrCb
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
	
def fourierDescriptor(cnt):# Fourier Decriptor used to extract feature from each figure(contour)
	max_length = 42
	cnt = cnt[:,0,:]
	cnt_complex = np.empty(cnt.shape[0],dtype=complex)
	#print(len(cnt_complex))
	cnt_complex.real = cnt[:,0]
	cnt_complex.imag = cnt[:,1]
	cnt_fft = np.fft.fft(cnt_complex)
	result = np.fft.fftshift(cnt_fft)
	low,high = int(len(cnt_complex)/2)-max_length/2,int(len(cnt_complex)/2)+max_length/2
	result = result[low:high]
	result = np.fft.ifftshift(result)
	return result

def image_process(image):
	skin, mask= skin_threshold(image)	
	gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
	dst = cv2.Laplacian(gray,cv2.CV_16S,ksize=3)
	Laplacian = cv2.convertScaleAbs(dst)
	h = cv2.findContours(Laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	contours = h[1] 
	areas = list()
	for i,cnt in enumerate(contours):
		areas.append(cv2.contourArea(cnt))
	ll = areas.index(max(areas))	
	cnt = contours[ll]
	fourier = fourierDescriptor(cnt)	
	return fourier
	
with open('/home/pi/New/train/final.pickle','rb') as f:
	clf_final=pickle.load(f)			
camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = 32
camera.hflip = False
camera.vflip = True
camera.rotation = 90
rawCapture = PiRGBArray(camera,size=(320,240))
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture,format = "bgr", use_video_port = True):
	image = frame.array
	image = cv2.bilateralFilter(image,9,75,75) #Filter
	cv2.imshow("Original",image)
	skin, mask= skin_threshold(image)
	cv2.imshow("MASK",mask)
	image2 = image.copy()
	gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
	dst = cv2.Laplacian(gray,cv2.CV_16S,ksize=3)
	Laplacian = cv2.convertScaleAbs(dst)
	h = cv2.findContours(Laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	contours = h[1] 
	areas = list()
	for i,cnt in enumerate(contours):
		areas.append(cv2.contourArea(cnt))
	ll = areas.index(max(areas))	
	cnt = contours[ll]
	result = fourierDescriptor(cnt)
	inputx = np.concatenate((result.real,result.imag))#
	inputx = inputx.reshape((1,inputx.shape[0]))#
	y=clf_final.predict(inputx)[0]
	
	
	
	ret = np.ones(image.shape,np.uint8)
	cv2.drawContours(ret,cnt,-1,(255,0,0),1)
	cv2.putText(ret,str(int(y)),(40,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),10)
	cv2.imshow("Contour",ret)

	key = cv2.waitKey(1) & 0xFF
	rawCapture.truncate(0)
	
	if key==ord("q"):
		break
	
