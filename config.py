class Config:
    EMG_PORT = "/dev/ttyUSB0"
    BAUDRATE = 115200
    CHANNELS = 10
    WINDOW_SAMPLES = 700
    XMODEL_PATHS = ["../compilation/compiled/emg_cnn.xmodel"]  # Add more for ensemble
    BUFFER_SIZE = 500
