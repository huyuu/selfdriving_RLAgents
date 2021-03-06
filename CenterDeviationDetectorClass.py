import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
import matplotlib.animation as animation
import os
import cv2


class CenterDeviationDetector():

    # Initializer

    def __init__(self):
        self.bottomAcceptableMargin = 30
        self.queueElementAmount = 3
        self.leftQueue = nu.ones(self.queueElementAmount, dtype=nu.float)
        self.rightQueue = nu.ones(self.queueElementAmount, dtype=nu.float)


    # MARK: - Public Methods

    def resetQueue(self):
        self.leftQueue = nu.ones(self.queueElementAmount, dtype=nu.float)
        self.rightQueue = nu.ones(self.queueElementAmount, dtype=nu.float)
        pass


    def getCenterDeviation(self, image_origin):
        y_start = 50
        height = 60
        y_end = y_start+height
        image_trimmed = image_origin[y_start: y_end, 0: 320]
        image_blurred = cv2.bilateralFilter(image_trimmed, 3, 100, 100)
        image_blurred[-self.bottomAcceptableMargin:, 80: 240, :] = cv2.bilateralFilter(image_blurred[-self.bottomAcceptableMargin:, 80: 240, :], 5, 200, 200)
        image_gray = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2GRAY)
        # image_trimmed = image_origin[y_start: y_end, 0: 320]
        # image_blurred = cv2.bilateralFilter(image_trimmed, 5, 200, 200)
        # image_gray = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2GRAY)
        # image_gray = cv2.cvtColor(image_trimmed,cv2.COLOR_RGB2GRAY)
        # dst = cv2.fastNlMeansDenoising(image_trimmed,h=20)
        # image_edge = cv2.Canny(dst, 150, 200)
        image_edge = cv2.Canny(image_gray, 140, 350)

        left = 20
        right = 300
        gap = None
        countours, hierarchy = cv2.findContours(image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for countour in countours:
            countour = countour[:, 0, :]
            xs_bottom = countour[countour[:, 1] > height - self.bottomAcceptableMargin, 0]
            # xs_bottom = countour[countour[:, 1] > height - self.bottomAcceptableMargin, 0]
            # xs_bottom = countour[countour[:, 1] > height - self.bottomAcceptableMargin, 0]
            if len(xs_bottom) == 0:
                continue
            possibleLefts = xs_bottom[xs_bottom < 160]
            if len(possibleLefts) > 0:
                possibleLeft = possibleLefts.max()
                left = max(possibleLeft, left)
            possibleRights = xs_bottom[xs_bottom > 160]
            if len(possibleRights) > 0:
                possibleRight = possibleRights.min()
                right = min(possibleRight, right)

        if right == 300:
            right = int(320-140)
        # right_normalized = (right - 160) / 160.0
        self.rightQueue = nu.insert(self.rightQueue, 0, right)[:-1]
        # print(f"rightQueue: {self.rightQueue}")
        if left == 20:
            left = 140
        # left_normalized = (160 - left) / 160.0
        self.leftQueue = nu.insert(self.leftQueue, 0, left)[:-1]

        leftQueue_mean = self.leftQueue.mean()
        rightQueue_mean = self.rightQueue.mean()
        roadCenter = (rightQueue_mean + leftQueue_mean)/2
        roadHalfWidth = (rightQueue_mean - leftQueue_mean)/2
        if abs(160 - roadCenter) >= 1e-2:
            gap = (160 - roadCenter) / roadHalfWidth
            return leftQueue_mean, rightQueue_mean, gap
        else:
            return leftQueue_mean, rightQueue_mean, None

        # for countour in countours:
        #     x = countour[0][0][0]
        #     y = countour[0][0][1]
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
        y_start = 50
        height = 60
        y_end = y_start+height
        image_trimmed = image_origin[y_start: y_end, 0: 320]
        image_blurred = cv2.bilateralFilter(image_trimmed, 3, 100, 100)
        # image_blurred = image_trimmed
        image_blurred[-self.bottomAcceptableMargin:, 80: 240, :] = cv2.bilateralFilter(image_blurred[-self.bottomAcceptableMargin:, 80: 240, :], 5, 200, 200)
        image_gray = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2GRAY)
        # image_trimmed = image_origin[y_start: y_end, 0: 320]
        # image_blurred = cv2.bilateralFilter(image_trimmed, 5, 200, 200)
        # image_gray = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2GRAY)
        # image_gray = cv2.cvtColor(image_trimmed,cv2.COLOR_RGB2GRAY)
        # dst = cv2.fastNlMeansDenoising(image_trimmed,h=20)
        # image_edge = cv2.Canny(dst, 150, 200)
        image_edge = cv2.Canny(image_gray, 140, 350)

        left = 20
        right = 300
        gap = None
        # lines = cv2.HoughLinesP(image_edge, rho=1, theta=nu.pi/360, threshold=50, minLineLength=50, maxLineGap=5)
        # if lines is not None:
        #     for line in lines:
        #         x1,y1,x2,y2 = line[0]
        #         if y1 > height - 20:
        #             if x1 < 160:
        #                 left = max(x1, left)
        #             else:
        #                 right = min(x1, right)
        #         if y2 > height - 20:
        #             if x2 < 160:
        #                 left = max(x2, left)
        #             else:
        #                 right = min(x2, right)
        #         # cv2.line(image_trimmed, (x1,y1), (x2,y2), (255, 0, 0), 2)
        #     if right == 320 and left == 0:
        #         return None, image_origin
        #     roadCenter = int((right + left)/2)
        #     roadHalfWidth = (right - left)/2
        #     gap = (160 - roadCenter) / roadHalfWidth
        #     cv2.circle(image_origin,(160,30+y_start),5,(255,0,0),thickness=1)
        #     cv2.circle(image_origin,(roadCenter,30+y_start),5,(0,0,255),thickness=1)
        #     return float(gap), image_origin
        # else:
        #     return None, image_origin
        countours, hierarchy = cv2.findContours(image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # countours = nu.array(countours)
        for countour in countours:
            countour = countour[:, 0, :]
            xs_bottom = countour[countour[:, 1] > height - self.bottomAcceptableMargin, 0]
            if len(xs_bottom) == 0:
                continue
            possibleLefts = xs_bottom[xs_bottom < 160]
            if len(possibleLefts) > 0:
                possibleLeft = possibleLefts.max()
                left = max(possibleLeft, left)
            possibleRights = xs_bottom[xs_bottom > 160]
            if len(possibleRights) > 0:
                possibleRight = possibleRights.min()
                right = min(possibleRight, right)

        if right == 300:
            right = int(320-140)
        # else:
        #     right_normalized = (right - 160) / 100
        self.rightQueue = nu.insert(self.rightQueue, 0, right)[:-1]
        if left == 20:
            left = 140
        # else:
        #     left_normalized = (160 - left) / 100.0
        self.leftQueue = nu.insert(self.leftQueue, 0, left)[:-1]
        # roadCenter = (right + left)/2
        # roadHalfWidth = (right - left)/2
        # gap = (160 - roadCenter) / roadHalfWidth
        # # Image center color_blue
        cv2.circle(image_edge, (int(self.leftQueue.mean()), height-self.bottomAcceptableMargin),5,(255,0,0),thickness=3)
        cv2.circle(image_edge, (int(self.rightQueue.mean()), height-self.bottomAcceptableMargin),5,(255,0,0),thickness=3)
        cv2.circle(image_edge, (int((self.rightQueue.mean()+self.leftQueue.mean())/2), height-self.bottomAcceptableMargin),5,(255,0,0),thickness=3)
        # The center of the road  color_red
        # cv2.circle(image_edge, (int(roadCenter),height-5),5,(255,0,0),thickness=3)
        cv2.circle(image_edge, (160,int(height-self.bottomAcceptableMargin)),5,(255,0,0),thickness=2)
        try:
            cv2.drawContours(image_edge, countours, 1, (255,0,0), 1)
        except:
            return image_edge
        return image_edge
        #
        # elif left == 0 and right != 320:
        #     right_normalized = (right - 160) / 100
        #     self.rightQueue = np.delete(self.rightQueue, -1, right)
        #     self.rightQueue = np.insert(self.rightQueue, 0, right)
        #     cv2.circle(image_edge, (int(right), height-self.bottomAcceptableMargin),5,(255,0,0),thickness=3)
        #     return None, self.rightQueue, image_edge
        #
        # elif left != 0 and right == 320:
        #     left_normalized = (160 - left) / 100.0
        #     self.leftQueue = np.delete(self.leftQueue, -1, left)
        #     self.leftQueue = np.insert(self.leftQueue, 0, left)
        #     cv2.circle(image_edge, (int(left), height-self.bottomAcceptableMargin),5,(255,0,0),thickness=3)
        #     return self.leftQueue, None, image_edge
        #
        # else:# right == 320 and left == 0
        #     return None, None, image_edge


    def showVedio(self):
        fig = pl.figure()
        plottingImages = []
        imagesDirPath = '../data/trace1/20210630_1759/IMG'
        for imageName in list(filter(lambda path: '.jpg' in path and 'center' in path, os.listdir(imagesDirPath))):
            imagePath = f"{imagesDirPath}/{imageName}"
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.getCenterDeviationWithImage(image)
            # _, _ = self.getCenterDeviation(image)
            image_plot = pl.imshow(image, cmap='gray')
            plottingImages.append([image_plot])
        ani = animation.ArtistAnimation(fig, plottingImages, interval=50)
        ani.save("center.gif")
        pl.show()



# MARK: - Main

if __name__ == '__main__':
    centerDeviationDetector = CenterDeviationDetector()
    centerDeviationDetector.showVedio()
