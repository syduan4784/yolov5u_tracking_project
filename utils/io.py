import csv, os

class TrajLogger:
    def __init__(self, path="outputs/tracks.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "id", "cls", "cx", "cy", "speed", "accel", "heading", "label"])

    def append(self, row):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
