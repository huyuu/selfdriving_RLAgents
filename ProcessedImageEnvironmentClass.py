import numpy as np
import queue
from datetime import datetime
import gym
from gym import spaces

from SimulatorDriverClass import SimulatorDriver
from CenterDeviationDetectorClassOrigin import CenterDeviationDetector


class ProcessedImageEnvironment(gym.Env):
    def __init__(self):
        super(ProcessedImageEnvironment, self).__init__()
        # for road detector
        self.centerDeviationDetector = CenterDeviationDetector()

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(int(self.centerDeviationDetector.queueElementAmount*2+3+1),))
        self.action_space = spaces.Discrete(4)
        self.observation_spec = {
            'shape': (int(self.centerDeviationDetector.queueElementAmount*2+3),), # delta from center line, speed (clipped to [0, 1]), steering angle, throttle
            'dtype': np.float,
            'minValue': -1.0,
            'maxValue': 1.0,
        }
        self.action_spec = {
            'shape': (4,), # left, right, straight, break
            'dtype': np.float,
            'minValue': 0.0,
            'maxValue': 1.0,
        }
        # for simulator driver
        self.imageFolderPath = './'
        self.simulatorDriver = SimulatorDriver()
        self.simulatorDriver.startServer()
        # last observation
        self.lastObservation = None
        # self.lastCenterDeviation = 0.0
        self.lastSteeringAngle = 0.0
        # disatisfied condition counters
        # self.centerDeviationUnobservableTimes = 0
        self.zeroSpeedTimes = 0
        self.continuousOffCourseTimes = 0


    def reset(self):
        # Todo: Implementation
        self.lastObservation = None
        self.lastCenterDeviation = 0.0
        self.lastSteeringAngle = 0.0
        # disatisfied condition counters
        # self.centerDeviationUnobservableTimes = 0
        self.zeroSpeedTimes = 0
        self.continuousOffCourseTimes = 0
        self.centerDeviationDetector.resetQueue()
        # restart simulatorDriver
        self.simulatorDriver.restart()
        steering_angle_after, throttle_after, speed_after, image_after = self.simulatorDriver.sendActionAndGetRawObservation(0.0, 0.01)
        leftQueue, rightQueue = self.centerDeviationDetector.getCenterDeviation(image_after)
        # if centerDeviation is None:
        #     centerDeviation = self.lastCenterDeviation
        # observation = np.array([centerDeviation, steering_angle_after, throttle_after, speed_after])
        observation = np.concatenate([leftQueue, rightQueue, np.array([steering_angle_after, throttle_after, speed_after])])
        self.lastObservation = observation
        # self.lastCenterDeviation = centerDeviation
        self.lastSteeringAngle = steering_angle_after
        # observation = np.zeros(4)
        return observation


    def step(self, action):
        observation = None
        reward = None
        isDone = False
        collideReward = -100
        # calculate action
        steering_angle_before, throttle_before = [0.0, 0.0]
        if action == 0: # turn left
            steering_angle_before, throttle_before = [-0.2, 1.0]
        elif action == 1: # turn right
            steering_angle_before, throttle_before = [0.2, 1.0]
        elif action == 2: # go straight
            steering_angle_before, throttle_before = [0.0, 1.0]
        else: # break
            steering_angle_before, throttle_before = [0.0, -0.5]
        # get observation
        try:
            steering_angle_after, throttle_after, speed_after, image_after = self.simulatorDriver.sendActionAndGetRawObservation(steering_angle_before, throttle_before)
        except queue.Empty as error:
            isDone = True
            return self.lastObservation, 0, True, None
        # calculate center deviation
        leftQueue, rightQueue = self.centerDeviationDetector.getCenterDeviation(image_after)
        # print(f"{centerDeviation}, {self.lastCenterDeviation}")
        # if centerDeviation is None:
        #     centerDeviation = self.lastCenterDeviation
        #     self.centerDeviationUnobservableTimes += 1
        # else:
        #     self.centerDeviationUnobservableTimes = 0
        if speed_after <= 1e-2:
            self.zeroSpeedTimes += 1
        else:
            self.zeroSpeedTimes = 0

        leftQueue_mean = leftQueue.mean()
        rightQueue_mean = rightQueue.mean()
        if leftQueue_mean <= 20.0/160.0 and rightQueue_mean <= 20.0/160.0:
            self.continuousOffCourseTimes += 1
        else:
            self.continuousOffCourseTimes = 0

        ratio = 0.0
        if leftQueue_mean >= 1e-3 and rightQueue_mean >= 1e-3:
            ratio = 1 - rightQueue_mean/leftQueue_mean
        elif leftQueue_mean >= 1e-3 and rightQueue_mean < 1e-3: # only left is observed
            ratio = leftQueue_mean/2.0
        elif leftQueue_mean < 1e-3 and rightQueue_mean >= 1e-3: # only right is observed
            ratio = rightQueue_mean/2.0
        # centerDeviation = 0.0
        # assert centerDeviation != None
        # observation = np.array([centerDeviation, steering_angle_after, throttle_after, speed_after])
        observation = np.concatenate([leftQueue, rightQueue, np.array([steering_angle_after, throttle_after, speed_after])])
        self.lastObservation = observation
        # self.lastCenterDeviation = centerDeviation
        self.lastSteeringAngle = steering_angle_after

        # calculate reward
        # reward = speed_after - abs(centerDeviation)
        if leftQueue_mean >= 1e-3 and rightQueue_mean >= 1e-3:
            if leftQueue_mean >= rightQueue_mean:
                reward = speed_after + rightQueue_mean/leftQueue_mean
            else:
                reward = speed_after + leftQueue_mean/rightQueue_mean
        elif leftQueue_mean >= 1e-3 and rightQueue_mean < 1e-3: # only left is observed
            reward = speed_after + leftQueue_mean/2.0
        elif leftQueue_mean < 1e-3 and rightQueue_mean >= 1e-3: # only right is observed
            reward = speed_after + rightQueue_mean/2.0
        else:
            reward = speed_after

        # check done
        # if self.centerDeviationUnobservableTimes >= 10:
        #     isDone = True
        #     reward = collideReward
        if self.continuousOffCourseTimes >= 5:
            print("offCourse!!!")
            isDone = True
            reward = collideReward
        elif self.zeroSpeedTimes >= 5:
            isDone = True
            reward = collideReward
        else:
            isDone = False

        # print(f"action = {action} -> obs = {observation}, reward = {reward}, isDone = {isDone}")
        return observation, reward, isDone, None


    def __calculateSteeringAngle(self, delta):
        newSteeringAngle = self.lastSteeringAngle + delta
        return max(-1.0, min(1.0, newSteeringAngle))


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
