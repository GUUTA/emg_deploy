import serial
import numpy as np
from config import Config


class EMGAcquisition:
    """
    EMG data acquisition from UART/Ethernet stream.
    Provides:
        - get_frame(n): returns n samples (used by onset detector)
        - get_window(): returns 700-sample window for CNN inference
    """

    def __init__(self,
                 port=Config.EMG_PORT,
                 baudrate=Config.BAUDRATE,
                 channels=Config.CHANNELS,
                 window_samples=Config.WINDOW_SAMPLES):

        self.channels = channels
        self.window_samples = window_samples

        # Serial port (for now; later upgrade to Ethernet UDP/TCP)
        self.ser = serial.Serial(port, baudrate, timeout=1)

    # -----------------------------------------------------------
    #         READ N SAMPLES FOR SHORT-FRAME ONSET DETECTOR
    # -----------------------------------------------------------
    def get_frame(self, n_samples):
        """
        Returns raw shape: (n_samples, channels)
        """
        buf = []
        while len(buf) < n_samples:
            line = self.ser.readline().decode().strip().split(',')
            if len(line) == self.channels:
                buf.append([float(x) for x in line])
        return np.array(buf)

    # -----------------------------------------------------------
    #        READ A FULL 700-SAMPLE INFERENCE WINDOW
    # -----------------------------------------------------------
    def get_window(self):
        """
        Returns shape: (1, window_samples, channels, 1)
        Prepared directly for FPGA CNN input.
        """
        buf = []
        while len(buf) < self.window_samples:
            line = self.ser.readline().decode().strip().split(',')
            if len(line) == self.channels:
                buf.append([float(x) for x in line])

        X = np.array(buf).reshape(1, self.window_samples, self.channels, 1)
        return X
