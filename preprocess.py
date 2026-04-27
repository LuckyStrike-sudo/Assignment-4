#This will contain the preprocess code that will process the video frame by frame
import numpy as np
import motion_detector
import cv2
def convert_to_grayscale(frame):
    return np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def preprocess(video_path, alpha, tau, delta, s, N, hit_threshold, bounce_threshold):
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        print("Error reading first frame")
        cap.release()
        return {}, {}, 0
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    ret, frame1 = cap.read()
    if not ret:
        print("Error reading second frame")
        cap.release()
        return {}, {}, 0
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    ret, frame2 = cap.read()
    if not ret:
        print("Error reading third frame")
        cap.release()
        return {}, {}, 0
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)


    frame0 = convert_to_grayscale(np.array(frame0))
    frame1 = convert_to_grayscale(np.array(frame1))
    frame2 = convert_to_grayscale(np.array(frame2))
    motion = motion_detector.motionDetector(frame0, frame1, frame2, alpha, tau, delta, s, N)
    results = {"frame": [], "objects": []}
    previous_objects = {}
    hit_cooldown = {}
    bounce_cooldown = {}
    hit_frames = []
    bounce_frames = []
    cooldown = 10
    i = 3
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = convert_to_grayscale(np.array(frame))
        tracking = motion.update(frame)

        objects = []
        for j, t in enumerate(tracking):
            objects.append({
                "id": t.objId,
                "x": t.x[0],
                "y": t.x[1],
                "bbox": t.bbox,
                "history": list(t.positions),
                "vx": t.x[2],
                "vy": t.x[3]
            })

            obj = t.objId
            vx = t.x[2]
            vy = t.x[3]
            if obj in previous_objects:
                prev_vx = previous_objects[obj]["vx"]
                prev_vy = previous_objects[obj]["vy"]
                if prev_vy > bounce_threshold and vy < -bounce_threshold and obj not in bounce_cooldown:
                    bounce_frames.append(i)
                    bounce_cooldown[obj] = cooldown
                if abs(vx - prev_vx) > hit_threshold and obj not in hit_cooldown:
                    hit_frames.append(i)
                    hit_cooldown[obj] = cooldown
            previous_objects[obj] = {"vx": vx, "vy": vy}
        for obj in list(hit_cooldown.keys()):
            hit_cooldown[obj] -= 1
            if hit_cooldown[obj] <= 0:
                del hit_cooldown[obj]
        for obj in list(bounce_cooldown.keys()):
            bounce_cooldown[obj] -= 1
            if bounce_cooldown[obj] <= 0:
                del bounce_cooldown[obj]
                
        

        results["frame"].append(i)
        results["objects"].append(objects)
        i += 1
    return results, {"hits": hit_frames, "bounces": bounce_frames}, frame_count