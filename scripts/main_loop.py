import signal
import sys
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import Config
from emg_capture import EMGAcquisition
from inference import FPGAEnsemble

emg_stream = queue.Queue(maxsize=Config.BUFFER_SIZE)
pred_stream = queue.Queue(maxsize=Config.BUFFER_SIZE)
running = True

def signal_handler(sig, frame):
    global running
    print("Shutting down...")
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def acquisition_loop():
    emg_device = EMGAcquisition()
    while running:
        window = emg_device.get_window()
        if not emg_stream.full():
            emg_stream.put(window)

def inference_loop():
    fpga = FPGAEnsemble()
    while running:
        if not emg_stream.empty():
            window = emg_stream.get()
            pred = fpga.predict(window)
            if not pred_stream.full():
                pred_stream.put(pred)

def start_gui():
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_title("EMG Signals")
    ax2.set_title("Predicted Class")
    line_objs = [ax1.plot([0]*Config.WINDOW_SAMPLES)[0] for _ in range(Config.CHANNELS)]
    text_obj = ax2.text(0.5, 0.5, "Class: -", fontsize=20, ha='center')

    def update(frame):
        if not emg_stream.empty():
            latest_window = emg_stream.queue[-1]
            for ch in range(Config.CHANNELS):
                line_objs[ch].set_ydata(latest_window[0,:,ch,0])
        if not pred_stream.empty():
            text_obj.set_text(f"Class: {pred_stream.queue[-1]}")
        return line_objs + [text_obj]

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
    plt.show()

if __name__ == "__main__":
    acq_thread = threading.Thread(target=acquisition_loop)
    inf_thread = threading.Thread(target=inference_loop)
    acq_thread.start()
    inf_thread.start()
    start_gui()
    acq_thread.join()
    inf_thread.join()
