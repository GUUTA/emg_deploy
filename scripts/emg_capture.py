import serial
import numpy as np
from config import Config

class EMGAcquisition:
    def __init__(self, port=Config.EMG_PORT, baudrate=Config.BAUDRATE,
                 channels=Config.CHANNELS, window_samples=Config.WINDOW_SAMPLES):
        self.ser = serial.Serial(port, baudrate)
        self.channels = channels
        self.window_samples = window_samples

    def get_window(self):
        buf = []
        while len(buf) < self.window_samples:
            line = self.ser.readline().decode().strip().split(',')
            if len(line) == self.channels:
                buf.append([float(x) for x in line])
        X = np.array(buf).reshape(1, self.window_samples, self.channels, 1)
        return X
