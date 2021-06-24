import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
import matplotlib.animation as animation
import os
import cv2


class CenterDeviationDetector():

    # Initializer

    def __init__(self):
        pass


    # MARK: - Public Methods

    def getCenterDeviation(self, image_origin):
        y_start = 60
        height = 50
        y_end = y_start+height
        image_trimmed = image_origin[y_start: y_end, 0: 320]
        image_blurred = cv2.bilateralFilter(image_trimmed, 5, 100, 100)
        image_gray = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2GRAY)
        # image_gray = cv2.cvtColor(image_trimmed,cv2.COLOR_RGB2GRAY)
        # dst = cv2.fastNlMeansDenoising(image_trimmed,h=20)
        # image_edge = cv2.Canny(dst, 150, 200)
        image_edge = cv2.Canny(image_gray, 150, 200)

        left = 0
        right = 320
        gap = None
        lines = cv2.HoughLinesP(image_edge, rho=1, theta=nu.pi/360, threshold=50, minLineLength=50, maxLineGap=5)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                if y1 > height - 20:
                    if x1 < 160:
                        left = max(x1, left)
                    else:
                        right = min(x1, right)
                if y2 > height - 20:
                    if x2 < 160:
                        left = max(x2, left)
                    else:
                        right = min(x2, right)
                # cv2.line(image_trimmed, (x1,y1), (x2,y2), (255, 0, 0), 2)
            if right == 320 and left == 0:
                return None
            roadCenter = int((right + left)/2)
            roadHalfWidth = (right - left)/2
            gap = (160 - roadCenter) / roadHalfWidth
            return float(gap)
        else:
            return None
        # cv2.circle(image_trimmed, (left, height-15),5,(255,0,0),thickness=-1)
        # cv2.circle(image_trimmed, (right, height-15),5,(255,0,0),thickness=-1)
        # cv2.circle(image_trimmed, (roadCenter, height-15),5,(0,255,0),thickness=-1)
        # # The center of the road  color_red
        # cv2.circle(image_trimmed, (160, height-15),5,(0,0,255),thickness=-1)
        # # return image_trimmed


        # countours, hierarchy = cv2.findContours(image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # # countours = nu.array(countours)
        # for countour in countours:
        #     countour = countour[:, 0, :]
        #     xs_bottom = countour[countour[:, 1] < 20, 0]
        #     if len(xs_bottom) == 0:
        #         continue
        #     possibleLefts = xs_bottom[xs_bottom < 160]
        #     if len(possibleLefts) > 0:
        #         possibleLeft = possibleLefts.max()
        #         left = max(possibleLeft, left)
        #     possibleRights = xs_bottom[xs_bottom > 160]
        #     if len(possibleRights) > 0:
        #         possibleRight = possibleRights.min()
        #         right = min(possibleRight, right)
        # roadCenter = (right + left)/2
        # roadHalfWidth = (right - left)/2
        # gap = (160 - roadCenter) / roadHalfWidth
        # # Image center color_blue
        # cv2.circle(image_trimmed, (left, 15),5,(255,0,0),thickness=-1)
        # cv2.circle(image_trimmed, (right, 15),5,(255,0,0),thickness=-1)
        # # The center of the road  color_red
        # cv2.circle(image_trimmed, (160,15),5,(0,0,255),thickness=-1)
        # return cv2.drawContours(image_trimmed, countours, -1, (255,0,0), 2)

        # for i in range(len(countours)):
        #     x = countours[i][0][0][0]
        #     y = countours[i][0][0][1]
        #
        #     if x < 100 and y > 20:
        #         left = x
        #         #continue
        #     elif x > 200 and y > 20:
        #         right = x
        #         #continue
        #     center = int((right-left)/2) + left
        #     Gap = (160-center)/center
        # return Gap


    # MAKR: - Private Methods

    def getCenterDeviationWithImage(self, image_origin):
        y_start = 60
        height = 50
        y_end = y_start+height
        image_trimmed = image_origin[y_start: y_end, 0: 320]
        image_blurred = cv2.bilateralFilter(image_trimmed, 5, 100, 100)
        image_gray = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2GRAY)
        # image_gray = cv2.cvtColor(image_trimmed,cv2.COLOR_RGB2GRAY)
        # dst = cv2.fastNlMeansDenoising(image_trimmed,h=20)
        # image_edge = cv2.Canny(dst, 150, 200)
        image_edge = cv2.Canny(image_gray, 150, 200)

        left = 0
        right = 320
        gap = None
        lines = cv2.HoughLinesP(image_edge, rho=1, theta=nu.pi/360, threshold=50, minLineLength=50, maxLineGap=5)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                if y1 > height - 20:
                    if x1 < 160:
                        left = max(x1, left)
                    else:
                        right = min(x1, right)
                if y2 > height - 20:
                    if x2 < 160:
                        left = max(x2, left)
                    else:
                        right = min(x2, right)
                # cv2.line(image_trimmed, (x1,y1), (x2,y2), (255, 0, 0), 2)
            if right == 320 and left == 0:
                return None, image_origin
            roadCenter = int((right + left)/2)
            roadHalfWidth = (right - left)/2
            gap = (160 - roadCenter) / roadHalfWidth
            cv2.circle(image_origin,(160,30+y_start),5,(255,0,0),thickness=1)
            cv2.circle(image_origin,(roadCenter,30+y_start),5,(0,0,255),thickness=1)
            return float(gap), image_origin
        else:
            return None, image_origin

    def showVedio(self):
        fig = pl.figure()
        plottingImages = []
        imagesDirPath = '../data/trace1/IMG'
        for imageName in list(filter(lambda path: '.jpg' in path and 'center' in path, os.listdir(imagesDirPath))):
            imagePath = f"{imagesDirPath}/{imageName}"
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gap, image = self.getCenterDeviationWithImage(image)
            if gap is None:
                pl.title(f"")
            else:
                pl.title(f'{gap}')
            image_plot = pl.imshow(image)
            plottingImages.append([image_plot])
        ani = animation.ArtistAnimation(fig, plottingImages, interval=50)
        pl.show()



# MARK: - Main

if __name__ == '__main__':
    centerDeviationDetector = CenterDeviationDetector()
    centerDeviationDetector.showVedio()
