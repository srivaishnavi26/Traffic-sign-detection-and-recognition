from pathlib import Path

def load_temporal_predictions(path):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            if "stable=" not in line:
                continue
            name = line.split()[0]
            stable = line.split("stable=")[1].split()[0]
            mapping[name] = stable
    return mapping
