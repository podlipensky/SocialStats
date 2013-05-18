import numpy as np
import cv2
import sys
# http://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python
def train():
    im = cv2.imread('sample/numbers/numbers.png')
    im3 = im.copy()

    # prepare image for further processing, remove noise
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    # find contours

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]

    # associate number with its graphical representation

    for cnt in contours:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)

            if  h>28:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                # work with current contour only
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                cv2.imshow('norm',im)

                # wait for user input on the picture - expect digit
                key = cv2.waitKey(0)

                if key == 27:
                    sys.exit()
                elif key in keys:
                    responses.append(int(chr(key)))
                    sample = roismall.reshape((1,100))
                    samples = np.append(samples,sample,0)
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)

    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    print "sampling complete"

    np.savetxt('sample/numbers/samples.data',samples)
    np.savetxt('sample/numbers/responses.data',responses)

def get_skeleton(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    ret,img = cv2.threshold(img, 230, 100, cv2.THRESH_BINARY_INV)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    cv2.imshow('skel',skel)
    # cv2.waitKey(0)
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
        cv2.imshow('skel',skel)
        # cv2.waitKey(0)

    return skel

# train()

# train data

samples = np.loadtxt('sample/numbers/samples.data',np.float32)
responses = np.loadtxt('sample/numbers/responses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)

im = cv2.imread('sample/numbers/256.png')
im = cv2.resize(im, (im.shape[0] * 4, im.shape[0] * 4))

out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
# thresh = gray
cv2.imshow('show', thresh)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = gray[y:y+h,x:x+w]
            roi = get_skeleton(roi)
            cv2.imshow('roi',roi)
            # cv2.waitKey(0)
            roismall = cv2.resize(roi,(10,10))

            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
            string = str(int((results[0][0])))
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

cv2.drawContours(im,contours,-1,(0,0,255),1)
# cv2.imshow('im',im)
cv2.imshow('out',out)
cv2.waitKey(0)