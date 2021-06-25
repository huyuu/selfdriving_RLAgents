import numpy as np
import pynput as pp
import queue
from datetime import datetime
import gym
from gym import spaces

from SimulatorDriverClass import SimulatorDriver


class RawImageEnvironment():
    def __init__(self):
        self.observation_spec = {
            'image': {
                'shape': (160, 320, 3), # delta from center line, speed (clipped to [0, 1]), steering angle
                'dtype': np.int32,
                'minValue': 0,
                'maxValue': 255
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
        self.observation = {}
        self.shouldExit = False
        # set listener
        def on_press(key):
            if key == pp.keyboard.Key.esc:
                self.shoueExit = True
        self.keyboardListener = pp.keyboard.Listener(
            on_press=on_press
        )
        self.keyboardListener.start()


    def reset(self):
        # Todo: Implementation
        self.shouldExit = False
        self.simulatorDriver.restart()
        steering_angle_after, throttle_after, speed_after, image_after = self.simulatorDriver.sendActionAndGetRawObservation(0.0, 0.01)
        observation = [
            image_after,
            np.array([steering_angle_after, throttle_after, speed_after])
        ]
        self.lastObservation = observation
        return observation


    def step(self, action):
        observation = None
        reward = None
        if self.shouldExit:
            return self.lastObservation, -100, True
        isDone = False
        # calculate action
        steering_angle_before, throttle_before = [0.0, 0.0]
        if action == 0: # turn left
            steering_angle_before, throttle_before = [-0.1, 1.0]
        elif action == 1: # turn right
            steering_angle_before, throttle_before = [0.1, 1.0]
        elif action == 2: # go straight
            steering_angle_before, throttle_before = [0.0, 1.0]
        else: # break
            steering_angle_before, throttle_before = [0.0, 0.0]
        # get observation
        try:
            steering_angle_after, throttle_after, speed_after, image_after = self.simulatorDriver.sendActionAndGetRawObservation(steering_angle_before, throttle_before)
        except queue.Empty as error:
            isDone = True
            return self.lastObservation, -100, True

        observation = [
            image_after,
            np.array([steering_angle_after, throttle_after, speed_after])
        ]
        self.lastObservation = observation

        if self.shouldExit:
            return self.lastObservation, -100, True
        # calculate reward
        reward = speed_after

        return observation, reward, isDone


if __name__ == '__main__':
    environment = Environment()
    print(environment.reset())
