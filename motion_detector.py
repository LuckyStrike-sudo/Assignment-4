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
            if r.area > 15 and r.area < 100:
                filtered.append(r)

        for r in filtered:
            temp = kalman_filter.kalmanFilter(r.centroid, self.nextId)
            self.potential.append(temp)
            self.nextId +=1

    def update(self, frame):
        self.previousThreeFrames = self.previousThreeFrames[1:] + [frame]
        self.frameCount += 1
        if len(self.tracking) != 0:
            for t in self.tracking:
                t.predict()
        if len(self.potential) != 0:
            for p in self.potential:
                p.predict()
        if self.frameCount % self.s != 0:
            for t in self.tracking:
                t.frames_since_update += 1
                if t.frames_since_update > self.alpha:
                    t.active = False
                self.tracking = [t for t in self.tracking if t.active]
            return
        
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
            if r.area > 15 and r.area < 100:
                filtered.append(r)
        centriods = [r.centroid for r in filtered]
        for t in self.tracking:
            for c in centriods:
                if np.linalg.norm(np.array(t.x[:2]) - np.array(c)) < self.delta:
                    t.update(np.array(c))
                    t.frames_since_update = 0
                    t.active = True
                    break