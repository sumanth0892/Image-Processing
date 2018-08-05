import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cv2

#Load an Image in Reading mode
img = cv2.imread('image.png',1)

#Display an image
cv2.imshow('image.png',img)

#Split an Image into its constituent parts

class Split_Equalize:
    def __init__(self):
        self.b,self.g,self.r = cv2.split(self)

    def hist(self):
        self.histred = cv2.calcHist([self],[2],[256],[0,256])
        self.histgreen = cv2.calcHist([self],[1],[256],[0,256])
        self.histblue = cv2.calcHist([self],[0],[256],[0,256])

        #Equalize the histogram
        self.rnew = cv2.equalizeHist(self.r)
        self.gnew = cv2.wqualizeHist(self.g)

    #Combine the three channels
    def merge(self.b,self.gnew,self.rnew):
        self.imagenew = cv2.merge((self.b,self.gnew,self.rnew))
        cv2.imwrite('newimage.png',self.imagenew)
        cv2.imshow(self.imagenew)

    #The above class takes a raw image and splits into its constituent
    #parts in colors and histograms.

class HSI_Stretching:
    def __init__(self):
        self.imghsl = cvtColor(self,cv2.COLOR_BGR2HSL)

    #Splitting components
    def split_equalize(self):
        self.h,self.s,self.l = cv2.split(self.imghsl)
        self.snew = cv2.equalizeHist(self.s)
        self.lnew = cv2.equalizeHist(self.l)
        self.newimagehsl = cv2.merge(self.h,self.snew,self.lnew)
        self.newimage = cv2.cvtColor(self.newimagehsl,cv2.COLOR_HSL2BGR)

class ZCA_Whitening:
    def __init__(self):
        self.newimage = cv2.imread(self,0)
        self.sigma = np.cov(self.newimage,rowvar = True)
        self.U,self.S,self.V = np.linalg.svd(self.sigma)
        epsilon = 1e-5
        self.ZCAmatrix = np.dot(self.U,np.dot(np.diag(1.0/np.sqrt(self.S+epsilon))))
        cv2.imwrite(ZCAmatrix,ZCAmatrix)


def main():
    img = cv2.imread('image.png',0)
    img = Split_Equalize(img)
    img = HSI_Stretching(img)
    img1 = ZCA_Whitening(img)
    img2 = cv2.imread(img)
    img21 = Split_Equalize(img2)
    
    
        
                               
