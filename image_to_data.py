from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import os
import glob
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

def fourierDescriptor(cnt):
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
	h = cv2.findContours(Laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	contours = h[1] 
	areas = list()
	for i,cnt in enumerate(contours):
		areas.append(cv2.contourArea(cnt))
	ll = areas.index(max(areas))	
	cnt = contours[ll]
	fourier = fourierDescriptor(cnt)	
	return fourier
	
	
folder0 = "/home/pi/New/number0/"
paths = glob.glob(os.path.join(folder0,'*.jpg'))
raw_data0 = []
for path in paths:
	
	img = cv2.imread(path)
	
	fd = image_process(img)
	raw_data0.append(fd)

raw_data0 = np.array(raw_data0)
np.save("/home/pi/New/train/data0",raw_data0)
y0 = np.empty(len(raw_data0))
y0[:] = 0
np.save("/home/pi/New/train/y0",y0)
	
folder1 = "/home/pi/New/number1/"
paths = glob.glob(os.path.join(folder1,'*.jpg'))
raw_data1 = []
for path in paths:
	img = cv2.imread(path)
	fd = image_process(img)
	raw_data1.append(fd)
raw_data1 = np.array(raw_data1)
np.save("/home/pi/New/train/data1",raw_data1)
y1 = np.empty(len(raw_data1))
y1[:] = 1
np.save("/home/pi/New/train/y1",y1)
	
folder2 = "/home/pi/New/number2/"
paths = glob.glob(os.path.join(folder2,'*.jpg'))
raw_data2 = []
for path in paths:
	img = cv2.imread(path)
	fd = image_process(img)
	raw_data2.append(fd)
raw_data2 = np.array(raw_data2)
np.save("/home/pi/New/train/data2",raw_data2)	
y2 = np.empty(len(raw_data2))
y2[:] = 2
np.save("/home/pi/New/train/y2",y2)
	
folder3 = "/home/pi/New/number3/"
paths = glob.glob(os.path.join(folder3,'*.jpg'))
raw_data3 = []
for path in paths:
	img = cv2.imread(path)
	fd = image_process(img)
	raw_data3.append(fd)
raw_data3 = np.array(raw_data3)
np.save("/home/pi/New/train/data3",raw_data3)
y3 = np.empty(len(raw_data3))
y3[:] = 3
np.save("/home/pi/New/train/y3",y3)
	
folder4 = "/home/pi/New/number4/"
paths = glob.glob(os.path.join(folder4,'*.jpg'))
raw_data4 = []
for path in paths:
	img = cv2.imread(path)
	fd = image_process(img)
	raw_data4.append(fd)
raw_data4 = np.array(raw_data4)
np.save("/home/pi/New/train/data4",raw_data4)
y4 = np.empty(len(raw_data4))
y4[:] = 4
np.save("/home/pi/New/train/y4",y4)

folder5 = "/home/pi/New/number5/"
paths = glob.glob(os.path.join(folder5,'*.jpg'))
raw_data5 = []
for path in paths:
	img = cv2.imread(path)
	fd = image_process(img)
	raw_data5.append(fd)
raw_data5 = np.array(raw_data5)
np.save("/home/pi/New/train/data5",raw_data5)
y5 = np.empty(len(raw_data5))
y5[:] = 5
np.save("/home/pi/New/train/y5",y5)

folder6 = "/home/pi/New/number6/"
paths = glob.glob(os.path.join(folder6,'*.jpg'))
raw_data6 = []
for path in paths:
	img = cv2.imread(path)
	fd = image_process(img)
	raw_data6.append(fd)
raw_data6 = np.array(raw_data6)
np.save("/home/pi/New/train/data6",raw_data6)
y6 = np.empty(len(raw_data6))
y6[:] = 6
np.save("/home/pi/New/train/y6",y6)

folder8 = "/home/pi/New/number8/"
paths = glob.glob(os.path.join(folder8,'*.jpg'))
raw_data8 = []
for path in paths:
	img = cv2.imread(path)
	fd = image_process(img)
	raw_data8.append(fd)
raw_data8 = np.array(raw_data8)
np.save("/home/pi/New/train/data8",raw_data8)
y8 = np.empty(len(raw_data8))
y8[:] = 8
np.save("/home/pi/New/train/y8",y8)

folder9 = "/home/pi/New/number9/"
paths = glob.glob(os.path.join(folder9,'*.jpg'))
raw_data9 = []
for path in paths:
	img = cv2.imread(path)
	fd = image_process(img)
	raw_data9.append(fd)
raw_data9 = np.array(raw_data9)
np.save("/home/pi/New/train/data9",raw_data9)
y9 = np.empty(len(raw_data9))
y9[:] = 9
np.save("/home/pi/New/train/y9",y9)
