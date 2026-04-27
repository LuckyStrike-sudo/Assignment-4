import numpy as np
from skimage.measure import label, regionprops
import skimage.morphology
import kalman_filter
#This contians the code for the motion detector class
class motionDetector:
    def __init__(self, frame0, frame1, frame2, alpha, tau, delta, s, N):
        self.alpha = alpha # frame hysyeresis for determining active or inactive objects
        self.tau = tau # Threshold for filtering out noise
        self.delta = delta # Distance threshold for determining if an object belongs to an object currltly being tracked
        self.s = s # Number of frames to skip between detections
        self.N = N # Maximum number of objext to track
        self.tracking = [] # List to store the currently tracked objects
        self.potential = [] # List to store the potential objects that are not currently being tracked
        self.previousThreeFrames = [frame0, frame1, frame2] # List to store the previous three frames for motion detection
        self.frameCount = 3 # Counter for the number of frames processed
        self.nextId = 0 # Counter for the next object ID to assign

        diff1 = np.abs(frame2.astype(np.int32) - frame1.astype(np.int32)) # Absolute difference between the current frame and the previous frame
        diff2 = np.abs(frame1.astype(np.int32) - frame0.astype(np.int32)) # Absolute difference between the previous frame and the frame before that
        motion = np.minimum(diff1, diff2) # Minimum of the two differences to reduce noise
        motion[motion < self.tau] = 0 # Set motion values below the threshold to 0
        motion[motion >= self.tau] = 1 # Set motion values above the threshold to 1
        dilated = skimage.morphology.dilation(motion, skimage.morphology.square(9)) # Dilate the motion mask to fill in gaps
        labels = label(dilated) # Label the connected components in the motion mask
        regions = regionprops(labels) # Get the properties of the labeled regions
        filtered = []
        for r in regions:
            minr, minc, maxr, maxc = r.bbox
            h = maxr - minr
            w = maxc - minc
            ratio = h/w
            if r.area > 50 and r.area < 400 and ratio > 0.5 and ratio < 2:
                filtered.append(r)

        for r in filtered:
            y, x = r.centroid
            temp = kalman_filter.kalmanFilter((x,y), self.nextId)
            self.potential.append(temp)
            self.nextId +=1
            temp.seen += 1
            temp.frames_since_update = 0
            temp.bbox = r.bbox

    def update(self, frame):
        self.previousThreeFrames = self.previousThreeFrames[1:] + [frame]
        self.frameCount += 1
        if len(self.tracking) != 0:
            for t in self.tracking:
                t.predict()
        if self.frameCount % self.s != 0:
            for t in self.tracking:
                t.frames_since_update += 1
                if t.frames_since_update > self.alpha:
                    t.active = False
            self.tracking = [t for t in self.tracking if t.active]
            for p in self.potential:
                p.frames_since_update += 1
                if p.frames_since_update > self.alpha:
                    p.active = False
            self.potential = [p for p in self.potential if p.active]
            return self.tracking
        
        diff1 = np.abs(self.previousThreeFrames[2].astype(np.int32) - self.previousThreeFrames[1].astype(np.int32)) # Absolute difference between the current frame and the previous frame
        diff2 = np.abs(self.previousThreeFrames[1].astype(np.int32) - self.previousThreeFrames[0].astype(np.int32)) # Absolute difference between the previous frame and the frame before that
        motion = np.minimum(diff1, diff2) # Minimum of the two differences to reduce noise
        motion[motion < self.tau] = 0 # Set motion values below the threshold to 0
        motion[motion >= self.tau] = 1 # Set motion values above the threshold to 1
        dilated = skimage.morphology.dilation(motion, skimage.morphology.square(9)) # Dilate the motion mask to fill in gaps
        labels = label(dilated) # Label the connected components in the motion mask
        regions = regionprops(labels) # Get the properties of the labeled regions
        filtered = []
        for r in regions:
            minr, minc, maxr, maxc = r.bbox
            h = maxr - minr
            w = maxc - minc
            ratio = h/w
            if r.area > 50 and r.area < 400 and ratio > 0.5 and ratio < 2:
                filtered.append(r)
        centriods = []
        for f in filtered:
            y, x = f.centroid
            centroid = (x,y)
            centriods.append({'centroid': centroid, 'bbox': f.bbox})
        for t in self.tracking:
            best = None
            for c in centriods:
                centriod = c["centroid"]
                diff = np.linalg.norm(np.array(centriod) - np.array((t.x[0], t.x[1])))
                if best is None or diff < best[0]:
                    best = (diff, c)
            if best is not None and best[0] < self.delta:
                match = best[1]
                t.update(np.array(match["centroid"]))
                t.bbox = match["bbox"]
                centriods.remove(best[1])
                t.frames_since_update = 0
            else:
                t.frames_since_update += 1
                if t.frames_since_update > self.alpha:
                    t.active = False
        self.tracking = [t for t in self.tracking if t.active]

        promoted = []
        for p in self.potential:
            best = None
            for c in centriods:
                diff = np.linalg.norm(np.array(c['centroid']) - np.array((p.x[0], p.x[1])))
                if best is None or diff < best[0]:
                    best = (diff, c)
            if best is not None and best[0] < self.delta:
                p.update(np.array(best[1]['centroid']))
                p.bbox = best[1]["bbox"]
                centriods.remove(best[1])
                p.frames_since_update = 0
                p.seen += 1
                if p.seen >= self.alpha:
                    if len(self.tracking) < self.N:
                        self.tracking.append(p)
                        promoted.append(p)
            else:
                p.frames_since_update += 1
                if p.frames_since_update > self.alpha:
                    p.active = False
        self.potential = [p for p in self.potential if p.active and p not in promoted]


        for c in centriods:
            temp = kalman_filter.kalmanFilter(c["centroid"], self.nextId)
            self.potential.append(temp)
            self.nextId +=1
            temp.seen += 1
            temp.frames_since_update = 0
            temp.bbox = c["bbox"]
        return self.tracking
