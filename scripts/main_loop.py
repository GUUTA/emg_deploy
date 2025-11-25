import signal
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import Config
from emg_capture import EMGAcquisition
from run_interface import FPGAEnsemble
from onset_detector import EMGOnsetDetector
import numpy as np

# ----------------------
# GLOBAL QUEUES
# ----------------------
emg_stream = queue.Queue(maxsize=Config.BUFFER_SIZE)
pred_stream = queue.Queue(maxsize=Config.BUFFER_SIZE)
onset_stream = queue.Queue(maxsize=Config.BUFFER_SIZE)
onset_flag = False  # For GUI
running = True

# ----------------------
# SIGNAL HANDLER
# ----------------------
def signal_handler(sig, frame):
    global running
    print("Shutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ----------------------
# EMG ACQUISITION THREAD
# ----------------------
def acquisition_loop():
    emg_device = EMGAcquisition()
    short_frame_samples = EMGOnsetDetector().short_frame_samples
    while running:
        # Full window for CNN inference
        window = emg_device.get_window()  # shape = (1, WINDOW_SAMPLES, CHANNELS, 1)
        if not emg_stream.full():
            emg_stream.put(window)

        # Short frame for onset detection
        frame = emg_device.get_frame(short_frame_samples)  # shape = (short_frame_samples, CHANNELS)
        if not onset_stream.full():
            onset_stream.put(frame)

# ----------------------
# ONSET DETECTION THREAD
# ----------------------
def onset_loop():
    global onset_flag
    detector = EMGOnsetDetector()
    while running:
        if not onset_stream.empty():
            frame = onset_stream.get()
            is_active = detector.detect(frame)
            onset_flag = is_active
            if is_active:
                print("Muscle contraction detected!")

# ----------------------
# FPGA INFERENCE THREAD (Optimized)
# ----------------------
def inference_loop(fpga: FPGAEnsemble):
    while running:
        if not emg_stream.empty():
            window = emg_stream.get()
            pred = fpga.predict(window)
            if not pred_stream.full():
                pred_stream.put(pred)

# ----------------------
# GUI THREAD
# ----------------------
def start_gui():
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_title("EMG Signals")
    ax2.set_title("Predicted Class & Onset")
    
    # one line per channel
    line_objs = [ax1.plot([0]*Config.WINDOW_SAMPLES)[0] for _ in range(Config.CHANNELS)]
    text_obj = ax2.text(0.5, 0.5, "Class: -  |  Contraction: NO", fontsize=16, ha='center')

    def update(frame):
        # Latest EMG window
        if not emg_stream.empty():
            latest_window = emg_stream.queue[-1].reshape(Config.WINDOW_SAMPLES, Config.CHANNELS)
            for ch in range(Config.CHANNELS):
                line_objs[ch].set_ydata(latest_window[:, ch])

            # Highlight onset in red if detected
            color = "red" if onset_flag else "blue"
            for line in line_objs:
                line.set_color(color)

        # Latest prediction and contraction status
        pred_text = "Class: -  |  Contraction: NO"
        if not pred_stream.empty():
            latest_pred = pred_stream.queue[-1]
            contraction_text = "YES" if onset_flag else "NO"
            pred_text = f"Class: {latest_pred}  |  Contraction: {contraction_text}"

        text_obj.set_text(pred_text)

        return line_objs + [text_obj]

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
    plt.show()

# ----------------------
# MAIN EXECUTION
# ----------------------
if __name__ == "__main__":
    # Load FPGA ensemble once (all models)
    fpga_ensemble = FPGAEnsemble(xmodel_paths=Config.XMODEL_PATHS)

    acq_thread = threading.Thread(target=acquisition_loop)
    onset_thread = threading.Thread(target=onset_loop)
    inf_thread = threading.Thread(target=inference_loop, args=(fpga_ensemble,))

    acq_thread.start()
    onset_thread.start()
    inf_thread.start()

    start_gui()

    acq_thread.join()
    onset_thread.join()
    inf_thread.join()
