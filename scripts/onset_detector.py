import numpy as np
from collections import deque
from config import Config


class EMGOnsetDetector:
    """
    EMG Onset–Offset Detector
    --------------------------
    - Learns baseline threshold from relaxed samples using RMS
    - Detects onset using RMS + stability voting
    - Designed for real-time streaming + FPGA inference
    """

    def __init__(
        self,
        baseline_samples=1000,
        short_frame_ms=50,
        onset_factor=3.0,
        hold_ms=30,
        fs=1000
    ):
        self.fs = fs
        self.short_frame_samples = int((short_frame_ms / 1000) * fs)
        self.hold_samples = int((hold_ms / 1000) * fs)

        self.channels = Config.CHANNELS

        self.baseline_buf = deque(maxlen=baseline_samples)
        self.recent_states = deque(maxlen=self.hold_samples)

        self.onset_factor = onset_factor
        self.threshold = None

    # -----------------------------------------------------------
    #               BASELINE THRESHOLD ESTIMATION
    # -----------------------------------------------------------
    def update_baseline(self, samples):
        """
        samples: 1D flatten data from a relaxed frame
        Instead of pushing raw samples, compute RMS so it matches detect() behavior.
        """
        samples = np.asarray(samples)

        # push RMS into baseline, not raw values
        rms_val = np.sqrt(np.mean(samples ** 2))
        self.baseline_buf.append(float(rms_val))

        if len(self.baseline_buf) == self.baseline_buf.maxlen:
            arr = np.array(self.baseline_buf)
            self.threshold = arr.mean() + self.onset_factor * arr.std()

    # -----------------------------------------------------------
    #                       RMS CALCULATION
    # -----------------------------------------------------------
    def compute_rms(self, window):
        w = np.array(window)

        if w.ndim == 2:
            rms_channels = np.sqrt(np.mean(w * w, axis=0))
            return np.mean(rms_channels)

        return np.sqrt(np.mean(w * w))

    # -----------------------------------------------------------
    #                 ONSET / OFFSET DETECTION
    # -----------------------------------------------------------
    def detect(self, frame):
        """
        frame shape expected: (short_frame_samples, CHANNELS)
        Returns True when muscle contraction onset is detected.
        """

        if self.threshold is None:
            flat = np.ravel(frame)
            self.update_baseline(flat)
            return False

        rms_val = self.compute_rms(frame)
        active = rms_val > self.threshold

        self.recent_states.append(active)

        if len(self.recent_states) < self.recent_states.maxlen:
            return False

        active_votes = sum(self.recent_states)
        return active_votes > (0.6 * len(self.recent_states))

    def reset(self):
        self.recent_states.clear()
