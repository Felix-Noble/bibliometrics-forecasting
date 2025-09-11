#/src/CustomCallBacks/CustomProgBar.py
import sys
import time
import tensorflow as tf
from collections import deque

class BatchTimeLogger(tf.keras.callbacks.Callback):
    def __init__(self, window=50):
        """
        Args:
            window (int): Number of recent batches to average over.
        """
        super().__init__()
        self.window = window
        self.times = deque(maxlen=window)
        self.last_time = None

    def on_train_batch_end(self, batch, logs=None):
        # Just collect timing data, no printing
        now = time.time()
        if self.last_time:
            self.times.append(now - self.last_time)

    def on_epoch_end(self, epoch, logs=None):
        if self.times:
            avg = sum(self.times) / len(self.times)
            print(f"Epoch {epoch+1}: avg batch time {avg:.3f}s")

class PatchProgbarCallback(tf.keras.callbacks.Callback):
    def __init__(self, update_freq=10, stateful_metrics=None):
        super().__init__()
        self.update_freq = update_freq
        self.stateful_metrics = stateful_metrics

    def on_train_begin(self, logs=None):
        for cb in self.model.callbacks:
            if isinstance(cb, tf.keras.callbacks.ProgbarLogger):
                cb.update_freq = self.update_freq
                if self.stateful_metrics:
                    cb.stateful_metrics = set(self.stateful_metrics)
                print(f"[Tuner] Patched ProgbarLogger: update_freq={cb.update_freq}, "
                      f"stateful_metrics={cb.stateful_metrics}")

class TQDMETAProgbar(tf.keras.callbacks.Callback):
    def __init__(self, update_every=10, total_steps=None):
        super().__init__()
        self.update_every = update_every
        self.total_steps = total_steps  # should be epochs * steps_per_epoch
        self.start_time = None
        self.global_step = 0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print("Training started...", flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.params['epochs']}", flush=True)
        self.epoch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.global_step += 1

        if self.global_step % self.update_every == 0:
            elapsed = time.time() - self.start_time
            steps_done = self.global_step

            # Compute ETA
            if self.total_steps:
                avg_step = elapsed / steps_done
                steps_remaining = self.total_steps - steps_done
                eta_sec = int(avg_step * steps_remaining)
                eta_min, eta_sec = divmod(eta_sec, 60)
                eta_str = f"{eta_min:02d}:{eta_sec:02d}"
                progress = f"{steps_done}/{self.total_steps}"
            else:
                eta_str = "?"
                progress = f"{steps_done}"

            # Build one-line progress bar style string
            msg = (
                f"\r[Step {progress}] "
                f"loss={logs.get('loss', 0.0):.4f} "
                f"ETA {eta_str}"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        # Make sure to overwrite the progress line with summary
        sys.stdout.write(" " * 80 + "\r")   # clear line
        sys.stdout.flush()
        print(
            f"Epoch {epoch+1} finished in {elapsed:.1f}s "
            f"- val_loss={logs.get('val_loss')} "
            f"- val_precision={logs.get('val_precision')}"
        )
class ETAProgbar(tf.keras.callbacks.Callback):
    def __init__(self, update_every=10, total_steps=None):
        super().__init__()
        self.update_every = update_every
        self.total_steps = total_steps  # optional, for better ETA
        self.start_time = None
        self.epoch_start_time = None
        self.global_step = 0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print("Training started...")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{self.params['epochs']}")

    def on_train_batch_end(self, batch, logs=None):
        self.global_step += 1
        if self.global_step % self.update_every == 0:
            elapsed = time.time() - self.start_time
            steps_done = self.global_step
            if self.total_steps:
                # ETA = avg_time_per_step * steps_remaining
                avg_time = elapsed / steps_done
                steps_remaining = self.total_steps - steps_done
                eta_sec = avg_time * steps_remaining
                eta_min = int(eta_sec // 60)
                eta_sec = int(eta_sec % 60)
                print(
                    f"Step {steps_done}/{self.total_steps} "
                    f"- ETA: {eta_min:02d}:{eta_sec:02d} "
                    f"- loss: {logs.get('loss', 0.0):.4f}"
                )
            else:
                print(
                    f"Step {steps_done} "
                    f"- elapsed {elapsed:.1f}s "
                    f"- loss: {logs.get('loss', 0.0):.4f}"
                )

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.1f}s. "
              f"val_loss: {logs.get('val_loss')}, "
              f"val_metric: {logs.get('val_precision')}")

