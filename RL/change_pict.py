import cv2
import sys

from numpy import *
from matplotlib.pyplot import *

class detectColor:
    def __init__(self):
        #TH:orange, white
        self.upper = array([[55, 255, 255], [255, 35, 255]])
        self.lower = array([[15, 50, 180], [0, 0, 200]])

    def setThreshold(self, upper, lower):
        self.upper = upper
        self.lower = lower

    def rgb2hsv(self, im):
        return array(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))

    def bgr2hsv(self, im):
        return array(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))

    def getMask(self, imIn):
        im = self.preprocessing(imIn)
        masks = zeros((self.upper.shape[0], im.shape[0], im.shape[1]))
        for l,u,i in zip(self.lower, self.upper, range(self.upper.shape[0])):
            masks[i,:,:] = array(cv2.inRange(im, l, u))
        return  masks

    def showThreshold(self):
        for l, u in zip(self.lower, self.upper):
            print(l, u)

    def getBin(self,im):
        masks = self.getMask(im)

        bin = zeros(im.shape[:2])
        for mask in masks:
            bin = cv2.bitwise_or(bin, self.morphology(mask))
        return bin

    def morphology(self, im):
        kernel = ones((5, 5), uint8)
        return cv2.erode(cv2.morphologyEx(cv2.dilate(im, kernel, iterations = 1), cv2.MORPH_OPEN, kernel), kernel, iterations = 1)

    def preprocessing(self, imIn):
        im = self.rgb2hsv(imIn)  #ここを変えるrgb2hsv,bgr2hsvのどちらかほかはopencvの公式ドキュメントへ
        im = cv2.bilateralFilter(im, 5, 40, 40)
        return im

    def getImg(self, im):
        masks = self.getMask(im)
        im = zeros(im.shape[:2], dtype = "float64")
        for mask, coef in zip(masks, range(1, masks.shape[0]+1)):
            im = im + self.morphology(mask)*coef
        return im/im.max()


def main():
    path = "test.jpg"

    im = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    dc = detectColor()

    #imshow(dc.preprocessing(im))
    #show()
    #sys.exit()


    dc.showThreshold()
    res = dc.getMask(im)
    bin = dc.getImg(im)

    clf()
    subplot(1,3,1)
    imshow(res[0])
    title("color 1")
    subplot(1,3,2)
    imshow(res[1])
    title("color 2")
    subplot(1,3,3)
    imshow(bin)
    title("or")
    colorbar()
    show()