"""Capture ~10k real DVXplorer events and save as /tmp/dvxplorer_replay.npy
for the prototype's 'replay' input mode."""
import numpy as np
import time
import dv_processing as dv

def main():
    cam = dv.io.camera.open()
    print(f"camera: {cam.getCameraName()} resolution: {cam.getEventResolution()}")
    target = 10000
    out = np.empty(target, dtype=[("x", "<i4"), ("y", "<i4"),
                                    ("p", "<i4"), ("t", "<i8")])
    n = 0
    t0 = time.monotonic()
    while n < target and time.monotonic() - t0 < 10.0:
        b = cam.getNextEventBatch()
        if b is None or b.size() == 0:
            time.sleep(0.001); continue
        arr = b.numpy()
        m = min(target - n, arr.shape[0])
        out["x"][n:n+m] = arr["x"][:m]
        out["y"][n:n+m] = arr["y"][:m]
        out["p"][n:n+m] = arr["polarity"][:m]
        out["t"][n:n+m] = arr["timestamp"][:m]
        n += m
    out = out[:n]
    np.save("/tmp/dvxplorer_replay.npy", out)
    print(f"saved {n} events to /tmp/dvxplorer_replay.npy in {time.monotonic()-t0:.2f}s")

if __name__ == "__main__":
    main()
