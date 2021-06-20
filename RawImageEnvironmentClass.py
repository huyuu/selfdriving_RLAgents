import numpy as np
from SimulatorDriverClass import SimulatorDriver


class RawImageEnvironment():
    def __init__(self):
        self.observation_spec = {
            'image': {
                'shape': (160, 320, 3), # delta from center line, speed (clipped to [0, 1]), steering angle
                'dtype': np.int32,
                'minValue': 0,
                'maxValue': 1
            },
            'speed': {
                'shape': (1,), # delta from center line, speed (clipped to [0, 1]), steering angle
                'dtype': np.float,
                'minValue': 0.0,
                'maxValue': 1.0
            },
            'steer angle': {
                'shape': (1,), # delta from center line, speed (clipped to [0, 1]), steering angle
                'dtype': np.float,
                'minValue': -1.0,
                'maxValue': 1.0
            }
        }
        self.action_spec = {
            'shape': (3,), # left, stop throttle (default always turns throttle on), right
            'dtype': np.float,
            'minValue': 0.0,
            'maxValue': 1.0,
        }
        self.simulatorDriver = SimulatorDriver()
        self.simulatorDriver.run()


    def reset(self):
        # Todo: Implementation
        observation = np.array((3, 3, 3))
        return observation


    def step(self, action):
        # Todo: Implementation
        return observation, reward, isDone, info


if __name__ == '__main__':
    environment = Environment()
    print(environment.reset())
