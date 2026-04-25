import numpy as np
import cv2
import skimage
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
        self.frameCount = 0 # Counter for the number of frames processed
        self.nextId = 0 # Counter for the next object ID to assign

        diff1 = abs(frame2 - frame1) # Absolute difference between the current frame and the previous frame
        diff2 = abs(frame1 - frame0) # Absolute difference between the previous frame and the frame before that
        motion = min(diff1, diff2) # Minimum of the two differences to reduce noise
        motion[motion < self.tau] = 0 # Set motion values below the threshold to 0
        motion[motion >= self.tau] = 1 # Set motion values above the threshold to 1
        dilated = skimage.morphology.dilation(motion, np.ones(9,9)) # Dilate the motion mask to fill in gaps
        labels = skimage.measure.label(dilated) # Label the connected components in the motion mask
        regions = skimage.measure.regionprops(labels) # Get the properties of the labeled regions


