import numpy as np
import queue
from datetime import datetime

from SimulatorDriverClass import SimulatorDriver
from CenterDeviationDetectorClass import CenterDeviationDetector


class ProcessedImageEnvironment():
    def __init__(self):
        self.observation_spec = {
            'shape': (4,), # delta from center line, speed (clipped to [0, 1]), steering angle
            'dtype': np.float,
            'minValue': -1.0,
            'maxValue': 1.0,
        }
        self.action_spec = {
            'shape': (3,), # left, break, right
            'dtype': np.float,
            'minValue': 0.0,
            'maxValue': 1.0,
        }
        # for simulator driver
        self.imageFolderPath = './'
        self.simulatorDriver = SimulatorDriver()
        self.simulatorDriver.startServer()
        # for road detector
        self.centerDeviationDetector = CenterDeviationDetector()
        # last observation
        self.lastObservation = None
        self.lastCenterDeviation = 0.0
        self.centerDeviationUnobservableTimes = 0
        self.zeroSpeedTimes = 0


    def reset(self):
        # Todo: Implementation
        self.lastCenterDeviation = 0.0
        self.centerDeviationUnobservableTimes = 0
        self.zeroSpeedTimes = 0
        self.simulatorDriver.restart()
        steering_angle_after, throttle_after, speed_after, image_after = self.simulatorDriver.sendActionAndGetRawObservation(0.0, 0.01)
        centerDeviation = self.centerDeviationDetector.getCenterDeviation(image_after)
        if centerDeviation is None:
            centerDeviation = self.lastCenterDeviation
        observation = np.array([centerDeviation, steering_angle_after, throttle_after, speed_after])

        # observation = np.zeros(4)
        self.lastCenterDeviation = observation
        return observation


    def step(self, action):
        observation = None
        reward = None
        isDone = False
        # calculate action
        steering_angle_before, throttle_before = [0.0, 0.0]
        if action == 0: # turn left
            steering_angle_before, throttle_before = [-0.1, 1.0]
        elif action == 1: # turn right
            steering_angle_before, throttle_before = [0.1, 1.0]
        else: # break
            steering_angle_before, throttle_before = [0.0, 0.0]
        # get observation
        try:
            steering_angle_after, throttle_after, speed_after, image_after = self.simulatorDriver.sendActionAndGetRawObservation(steering_angle_before, throttle_before)
        except queue.Empty as error:
            isDone = True
            return self.lastObservation, -10, True
        # calculate center deviation
        centerDeviation = self.centerDeviationDetector.getCenterDeviation(image_after)
        # print(f"{centerDeviation}, {self.lastCenterDeviation}")
        if centerDeviation is None:
            centerDeviation = self.lastCenterDeviation
            self.centerDeviationUnobservableTimes += 1
        else:
            self.centerDeviationUnobservableTimes = 0
        if speed_after <= 1e-3:
            self.zeroSpeedTimes += 1
        else:
            self.zeroSpeedTimes = 0
        # centerDeviation = 0.0
        observation = np.array([centerDeviation, steering_angle_after, throttle_after, speed_after])
        self.lastObservation = observation
        self.lastCenterDeviation = centerDeviation

        # calculate reward
        reward = speed_after - np.abs(centerDeviation)

        # check done
        if self.centerDeviationUnobservableTimes >= 10:
            isDone = True
        elif self.zeroSpeedTimes >= 5:
            isDone = True
            reward = -10

        # print(f"action = {action} -> obs = {observation}, reward = {reward}, isDone = {isDone}")
        return observation, reward, isDone


    # MARK: - Private Methods



if __name__ == '__main__':
    env = ProcessedImageEnvironment()
    for episode in range(10):
        observation = env.reset()
        reward = 0.0
        isDone = False
        action = 0
        while not isDone:
            action = np.random.choice([0, 1, 2])
            _start = datetime.now()
            observation, reward, isDone = env.step(action)
            # print(f"costs: {(datetime.now()-_start).total_seconds()}")
            print(f"action = {action} -> obs = {observation}, reward = {reward}, isDone = {isDone}")
        print(f"episode {episode+1}: reward = {reward}")
