def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter)


class IoUTracker:
    def __init__(self, thresh=0.5):
        self.thresh = thresh
        self.tracks = {}
        self.next_id = 0

    def assign_id(self, box):
        for tid, prev_box in self.tracks.items():
            if iou(box, prev_box) > self.thresh:
                self.tracks[tid] = box
                return tid

        tid = self.next_id
        self.tracks[tid] = box
        self.next_id += 1
        return tid

    def reset(self):
        self.tracks.clear()
        self.next_id = 0
