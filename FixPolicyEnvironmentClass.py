import numpy as np
import queue
from datetime import datetime
import gym
from gym import spaces

from SimulatorDriverClass import SimulatorDriver
from CenterDeviationDetectorClass import CenterDeviationDetector


class FixPolicyEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FixPolicyEnvironment, self).__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
        self.action_space = spaces.Discrete(4)
        self.observation_spec = {
            'shape': (4,), # delta from center line, speed (clipped to [0, 1]), steering angle, throttle
            'dtype': np.float,
            'minValue': -1.0,
            'maxValue': 1.0,
        }
        self.action_spec = {
            'shape': (4,), # left, right, straight, break
            'dtype': np.int,
            'minValue': 0.0,
            'maxValue': 1.0,
        }
        # for simulator driver
        self.imageFolderPath = './'
        # self.simulatorDriver = SimulatorDriver()
        # self.simulatorDriver.startServer()
        # for road detector
        self.centerDeviationDetector = CenterDeviationDetector()
        # last observation
        self.lastObservation = None
        self.lastCenterDeviation = 0.0
        self.centerDeviationUnobservableTimes = 0
        self.zeroSpeedTimes = 0
        self.continuousOffCourseTimes = 0

        self.lastThrottleLevel = 0.0
        self.lastSpeed = 0.0
        self.lastSteeringAngle = 0.0

        self.step_count = 0


    def _reset(self):
        # Todo: Implementation
        self.lastCenterDeviation = 0.0
        self.centerDeviationUnobservableTimes = 0
        self.lastCenterDeviation = 0.0
        self.zeroSpeedTimes = 0
        self.continuousOffCourseTimes = 0
        self.lastThrottleLevel = 0.0
        self.lastSpeed = 0.0
        self.lastSteeringAngle = 0.0
        self.step_count = 0
        # self.simulatorDriver.restart()
        # steering_angle_after, throttle_after, speed_after, image_after = self.simulatorDriver.sendActionAndGetRawObservation(0.0, 0.01)
        # centerDeviation = self.centerDeviationDetector.getCenterDeviation(image_after)
        # if centerDeviation is None:
        #     centerDeviation = self.lastCenterDeviation
        # observation = np.array([centerDeviation, steering_angle_after, throttle_after, speed_after])
        # self.lastObservation = observation
        # self.lastCenterDeviation = centerDeviation

        observation = np.zeros(4, dtype=np.float32)
        return observation


    def _step(self, action):
        observation = None
        reward = None
        isDone = False
        collideReward = -50
        self.step_count += 1
        # calculate action
        steering_angle_before, throttle_before = [0.0, 0.0]
        if action == 0: # turn left
            steering_angle_before, throttle_before = [-0.3, 1.0]
        elif action == 1: # turn right
            steering_angle_before, throttle_before = [0.3, 1.0]
        elif action == 2: # go straight
            steering_angle_before, throttle_before = [0.0, 1.0]
        else: # break
            steering_angle_before, throttle_before = [0.0, 0.0]

        # calculate reward
        myAtion = 0
        if abs(self.lastCenterDeviation) <= 0.2:
            myAction = 2
        elif self.lastCenterDeviation > 0.2:
            myAction = 0
        else:
            myAction = 1

        if action == myAction:
            reward = 1
        else:
            reward = -1

        # # get observation
        # try:
        #     steering_angle_after, throttle_after, speed_after, image_after = self.simulatorDriver.sendActionAndGetRawObservation(steering_angle_before, throttle_before)
        # except queue.Empty as error:
        #     isDone = True
        #     return self.lastObservation, -1, True
        # # calculate center deviation
        # centerDeviation = self.centerDeviationDetector.getCenterDeviation(image_after)

        centerDeviation = np.random.normal(loc=0.0, scale=0.4)
        # choosedAction = np.random.choice([0, 1, 2])
        # centerDeviation = None
        # if choosedAction == 0:
        #     centerDeviation = (1.0 - 0.2) * np.random.rand() + 0.2
        # elif choosedAction == 1:
        #     centerDeviation = (0.2+0.2) * np.random.rand() - 0.2
        # else:
        #     centerDeviation = (-0.2 + 1.0) * np.random.rand() - 1.0
        # centerDeviation = np.random.rand()
        self.lastThrottleLevel = max(0.0, min(1.0, self.lastThrottleLevel + throttle_before))
        if self.lastThrottleLevel >= 0.5:
            self.lastSpeed += 0.2
        else:
            self.lastSpeed -= 0.1
        self.lastSpeed = max(0.0, min(1.0, self.lastSpeed))
        self.lastSteeringAngle = max(-1.0, min(1.0, self.lastSteeringAngle + steering_angle_before))

        # centerDeviation = 0.0
        # assert centerDeviation != None
        observation = np.array([centerDeviation, self.lastSteeringAngle, self.lastThrottleLevel, self.lastSpeed], dtype=np.float32)
        self.lastObservation = observation
        self.lastCenterDeviation = centerDeviation

        # check done
        if self.step_count >= 1000:
            isDone = True
        else:
            isDone = False

        # print(f"action = {action} -> obs = {observation}, reward = {reward}, isDone = {isDone}")
        return observation, float(reward), isDone, myAction


    def _close(self):
        pass


    def _render(self, mode='human'):
        pass


    def _seed(self, seed=None):
        self.id = np.random()


    def get_action_meanings(self):
        return ['turn left', 'turn right', 'go straight', 'break']


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
