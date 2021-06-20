#parsing command line arguments
import argparse
#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking
import eventlet
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO
#load our saved model
# from keras.models import load_model
import pynput as pp
from time import sleep
import multiprocessing as mp
import queue
from sys import platform


class SimulatorDriver():

    # MARK: - Initializer

    def __init__(self):
        # https://pypi.org/project/pynput/
        self.keyboard = pp.keyboard.Controller()
        self.mouse = pp.mouse.Controller()
        # https://stackoverflow.com/questions/8220108/how-do-i-check-the-operating-system-in-python/46629115
        self.mouseStartPosition = None
        if platform == 'linux' or platform == 'linux2':
            self.mouseStartPosition = (972, 684)
        elif platform == 'win32':
            self.mouseStartPosition = (640, 451)
        elif platform == 'darwin':
            self.mouseStartPosition = (640, 451)
        else:
            self.mouseStartPosition = (640, 451)

        self.actionQueue = mp.Queue(maxsize=1)
        self.observationQueue = mp.Queue(maxsize=1)

        # self.throttle = mp.Value('d', float(0.0))
        # self.steering_angle = mp.Value('d', float(0.0))
        # self.speed = mp.Value('d', float(0.0))


    # MARK: - Public Methods

    def startServer(self):
        process = mp.Process(target=helper_startServer, args=(self.actionQueue, self.observationQueue))
        process.start()
        sleep(10)


    def restart(self):
        # Press and release esc
        self.keyboard.press(pp.keyboard.Key.esc)
        self.keyboard.release(pp.keyboard.Key.esc)
        # mouse click
        self.mouse.position = self.mouseStartPosition
        self.mouse.press(pp.mouse.Button.left)
        self.mouse.release(pp.mouse.Button.left)
        # get first observation
        observation = 0
        for i in range(10):
            self.sendActionAndGetRawObservation(0.0, 1.0)


    def sendActionAndGetRawObservation(self, steering_angle, throttle):
        self.__sendAction(steering_angle, throttle)
        observation = self.__getObservation()
        # print(f"action = ({steering_angle}, {throttle}); observation = ({observation})")
        return observation

    def sendActionWithoutFeedback(self, steering_angle, throttle):
        self.__sendAction(steering_angle, throttle)


    # MARK: - Private Methods

    def __sendAction(self, steering_angle, throttle):
        # self.throttle = throttle
        # self.steering_angle = steering_angle
        # while not self.actionQueue.empty():
        #     sleep(0.001)
        if self.actionQueue.full():
            _ = self.actionQueue.get()
        self.actionQueue.put((steering_angle, throttle), block=True)


    def __getObservation(self):
        # while self.observationQueue.empty():
        #     sleep(0.001)
        # print("__; getting observation ...")
        observation = self.observationQueue.get(block=True, timeout=5)
        # print(f"__; got observation.")
        return observation


# MARK: - Helper Async Functions

def helper_startServer(actionQueue, observationQueue):
    sio = socketio.Server()

    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': steering_angle.__str__(),
                'throttle': throttle.__str__()
            },
            skip_sid=True)

    #registering event handler for the server
    @sio.on('telemetry')
    def telemetry(sid, data):
        # check if isActionSet
        # while not isActionSet.is_set():
        #     sleep(0.01)

        if data:
            # print(f"telemetry: {datetime.now()}, data: {data.keys}")
            if not actionQueue.empty():
                # get action from actionQueue
                steering_angle_before, throttle_before = actionQueue.get(block=True)
                # send action to simulator
                send_control(steering_angle_before, throttle_before)
            # get observation from simulator
            steering_angle_after = float(data["steering_angle"])
            throttle_after = float(data["throttle"]) / 25.0
            speed_after = float(data["speed"]) / 30.5
            image_after = np.asarray(Image.open(BytesIO(base64.b64decode(data["image"]))))
            # put observation to observationQueue
            if observationQueue.full():
                _ = observationQueue.get(block=True)
            observationQueue.put((steering_angle_after, throttle_after, speed_after, image_after), block=True)

            # print(f"telemetry: {datetime.now()}, data: {data}")
            # # The current steering angle of the car
            # steering_angle = float(data["steering_angle"])
            # # The current throttle of the car, how hard to push peddle
            # throttle = float(data["throttle"])
            # # The current speed of the car
            # speed = float(data["speed"])
            # # The current image from the center camera of the car
            # image = Image.open(BytesIO(base64.b64decode(data["image"])))
            # send_control(0.5, 0.5)
            # try:
            #     image = np.asarray(image)       # from PIL image to numpy array
            #     image = utils.preprocess(image) # apply the preprocessing
            #     image = np.array([image])       # the model expects 4D array
            #
            #     # predict the steering angle for the image
            #     steering_angle = float(model.predict(image, batch_size=1))
            #     # lower the throttle as the speed increases
            #     # if the speed is above the current speed limit, we are on a downhill.
            #     # make sure we slow down first and then go back to the original max speed.
            #     global speed_limit
            #     if speed > speed_limit:
            #         speed_limit = MIN_SPEED  # slow down
            #     else:
            #         speed_limit = MAX_SPEED
            #     throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            #
            #     print('{} {} {}'.format(steering_angle, throttle, speed))
            #     send_control(steering_angle, throttle)
            # except Exception as e:
            #     print(e)
            #
            # # save frame
            # if args.image_folder != '':
            #     timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            #     image_filename = os.path.join(args.image_folder, timestamp)
            #     image.save('{}.jpg'.format(image_filename))
        else:
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        # print("connected")
        send_control(0, 0)

    @sio.on('disconnect')
    def disconnect(sid):
        # print("disconnect", sid)
        pass

    #our flask (web) app
    app = Flask(__name__)
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


if __name__ == '__main__':
    simulatorDriver = SimulatorDriver()
    simulatorDriver.startServer()
    simulatorDriver.restart()
    for i in range(100):
        print(i)
        if i % 2 == 0:
            simulatorDriver.sendActionAndGetRawObservation(0.1, 1.0)
        else:
            simulatorDriver.sendActionAndGetRawObservation(-0.1, 1.0)
