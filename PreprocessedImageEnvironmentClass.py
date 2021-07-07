import numpy as np
import pynput as pp
import queue
from datetime import datetime
import gym
from gym import spaces
import cv2

from SimulatorDriverClass import SimulatorDriver


class PreprocessedImageEnvironment():
    def __init__(self):
        self.observation_spec = {
            'image': {
                'shape': (60, 320, 1), # delta from center line, speed (clipped to [0, 1]), steering angle
                'dtype': np.float,
                'minValue': 0,
                'maxValue': 1.0
            },
            'subPara': {
                'shape': (3,), # delta from center line, speed (clipped to [0, 1]), steering angle
                'dtype': np.float,
                'minValue': -1.0,
                'maxValue': 1.0
            },
            # 'steer angle': {
            #     'shape': (1,), # delta from center line, speed (clipped to [0, 1]), steering angle
            #     'dtype': np.float,
            #     'minValue': -1.0,
            #     'maxValue': 1.0
            # },
            # 'throttle_level': {
            #     'shape': (1,), # delta from center line, speed (clipped to [0, 1]), steering angle
            #     'dtype': np.float,
            #     'minValue': 0.0,
            #     'maxValue': 1.0
            # }
        }
        self.action_spec = {
            'shape': (4,), # left, stop throttle (default always turns throttle on), right
            'dtype': np.int
        }
        self.simulatorDriver = SimulatorDriver()
        self.simulatorDriver.startServer()
        self.shouldExit = False
        # set listener
        def on_press(key):
            if key == pp.keyboard.Key.esc:
                self.shoueExit = True
        self.keyboardListener = pp.keyboard.Listener(
            on_press=on_press
        )
        self.keyboardListener.start()
        # last properties
        self.lastObservation = None
        self.lastSpeed = 0.0

        self.collideReward = -30


    def reset(self):
        # Todo: Implementation
        self.shouldExit = False
        self.simulatorDriver.restart()
        steering_angle_after, throttle_after, speed_after, image_after = self.simulatorDriver.sendActionAndGetRawObservation(0.0, 0.01)
        image_after = self.__getEdgeImage(image_after)
        observation = [
            (image_after/255.0).astype(np.float),
            np.array([steering_angle_after, throttle_after, speed_after])
        ]
        self.lastObservation = observation
        self.lastSpeed = speed_after
        return observation


    def step(self, action):
        observation = None
        reward = None
        if self.shouldExit:
            return self.lastObservation, self.collideReward, True, None
        isDone = False
        # calculate action
        steering_angle_before, throttle_before = [0.0, 0.0]
        if action == 0: # turn left
            steering_angle_before, throttle_before = [-0.35, 1.0]
        elif action == 1: # turn right
            steering_angle_before, throttle_before = [0.35, 1.0]
        elif action == 2: # go straight
            steering_angle_before, throttle_before = [0.0, 1.0]
        else: # break
                steering_angle_before, throttle_before = [0.0, -0.5]
        # get observation
        try:
            steering_angle_after, throttle_after, speed_after, image_after = self.simulatorDriver.sendActionAndGetRawObservation(steering_angle_before, throttle_before)
        except queue.Empty as error:
            isDone = True
            return self.lastObservation, self.collideReward, True, None

        image_after = self.__getEdgeImage(image_after)
        observation = [
            (image_after/255.0).astype(np.float),
            np.array([steering_angle_after, throttle_after, speed_after], dtype=np.float)
        ]
        self.lastObservation = observation
        self.lastSpeed = speed_after

        if self.shouldExit:
            return self.lastObservation, self.collideReward, True, None
        # calculate reward
        reward = speed_after

        return observation, reward, isDone, None


    def __getEdgeImage(self, image_origin, shouldTrim=True):
        bottomAcceptableMargin = 30
        y_start = 50 if shouldTrim else 0
        height = 60 if shouldTrim else 120
        y_end = y_start+height
        image_trimmed = image_origin[y_start: y_end, 0: 320]
        image_blurred = cv2.bilateralFilter(image_trimmed, 3, 70, 70)
        image_blurred[-bottomAcceptableMargin:, 80: 240, :] = cv2.bilateralFilter(image_blurred[-bottomAcceptableMargin:, 80: 240, :], 5, 200, 200)
        image_gray = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2GRAY)
        # image_gray = cv2.cvtColor(image_trimmed,cv2.COLOR_RGB2GRAY)
        # dst = cv2.fastNlMeansDenoising(image_trimmed,h=20)
        # image_edge = cv2.Canny(dst, 150, 200)
        image_edge = cv2.Canny(image_gray, 150, 350)
        return image_edge

if __name__ == '__main__':
    environment = Environment()
    print(environment.reset())
