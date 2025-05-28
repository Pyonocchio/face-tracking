import os
import blobconverter
import cv2
import depthai as dai
import numpy as np
import datetime
from MultiMsgSync import TwoStageHostSeqSync

VIDEO_SIZE = (1072, 1072)
EYE_STATE_CLASSES = ['closed-eyes', 'open-eyes']

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 2, self.line_type)

class FaceRecognition:
    def __init__(self, db_path) -> None:
        self.read_db(db_path)

    def cosine_distance(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b.T) / (a_norm * b_norm)

    def new_recognition(self, results):
        max_, label_ = 0, "UNKNOWN"
        for label in self.labels:
            for j in self.db_dic.get(label):
                conf = self.cosine_distance(j, results)
                if conf > max_:
                    max_ = conf
                    label_ = label
        if max_ >= 0.5:
            return max_, label_
        else:
            return 1 - max_, "UNKNOWN"

    def read_db(self, databases_path):
        self.labels = []
        self.db_dic = {}
        for file in os.listdir(databases_path):
            if file.endswith(".npz"):
                label = os.path.splitext(file)[0]
                self.labels.append(label)
                with np.load(os.path.join(databases_path, file)) as db:
                    self.db_dic[label] = [db[j] for j in db.files]

def notify_parent(child_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ALERT] {timestamp} - {child_name} has opened their eyes! Notify the parent.")

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

print("Creating pipeline...")
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 640)
cam.setVideoSize(VIDEO_SIZE)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

xout_cam = pipeline.create(dai.node.XLinkOut)
xout_cam.setStreamName("color")
cam.video.link(xout_cam.input)

copy_manip = pipeline.create(dai.node.ImageManip)
cam.preview.link(copy_manip.inputImage)
copy_manip.setNumFramesPool(20)
copy_manip.setMaxOutputFrameSize(3 * 640 * 640)

face_det_manip = pipeline.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
copy_manip.out.link(face_det_manip.inputImage)

face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
face_det_nn.setConfidenceThreshold(0.5)
face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
face_det_manip.out.link(face_det_nn.input)

face_det_xout = pipeline.create(dai.node.XLinkOut)
face_det_xout.setStreamName("detection")
face_det_nn.out.link(face_det_xout.input)

# Headpose
headpose_manip = pipeline.create(dai.node.ImageManip)
headpose_manip.initialConfig.setResize(60, 60)
headpose_manip.inputConfig.setWaitForMessage(False)
copy_manip.out.link(headpose_manip.inputImage)

headpose_nn = pipeline.create(dai.node.NeuralNetwork)
headpose_nn.setBlobPath(blobconverter.from_zoo(name="head-pose-estimation-adas-0001", shaves=6))
headpose_manip.out.link(headpose_nn.input)

# Face Recognition
face_rec_manip = pipeline.create(dai.node.ImageManip)
face_rec_manip.initialConfig.setResize(112, 112)
face_rec_manip.inputConfig.setWaitForMessage(False)
copy_manip.out.link(face_rec_manip.inputImage)

face_rec_nn = pipeline.create(dai.node.NeuralNetwork)
face_rec_nn.setBlobPath(blobconverter.from_zoo(name="face-recognition-arcface-112x112", zoo_type="depthai", shaves=6))
face_rec_manip.out.link(face_rec_nn.input)

arc_xout = pipeline.create(dai.node.XLinkOut)
arc_xout.setStreamName("recognition")
face_rec_nn.out.link(arc_xout.input)

# Eye classification (custom trained model)
eye_manip = pipeline.create(dai.node.ImageManip)
eye_manip.initialConfig.setResize(640, 640)
eye_manip.setMaxOutputFrameSize(3 * 640 * 640)
eye_manip.inputConfig.setWaitForMessage(False)
copy_manip.out.link(eye_manip.inputImage)

eye_nn = pipeline.create(dai.node.NeuralNetwork)
eye_nn.setBlobPath("pyono-results/result/best_openvino_2022.1_6shave.blob")
eye_manip.out.link(eye_nn.input)

eye_xout = pipeline.create(dai.node.XLinkOut)
eye_xout.setStreamName("eye")
eye_nn.out.link(eye_xout.input)

with dai.Device(pipeline) as device:
    facerec = FaceRecognition("databases")
    sync = TwoStageHostSeqSync(include_eye=True)
    text = TextHelper()

    queues = {}
    for name in ["color", "detection", "recognition", "eye"]:
        queues[name] = device.getOutputQueue(name)

    while True:
        for name, q in queues.items():
            if q.has():
                sync.add_msg(q.get(), name)

        msgs = sync.get_msgs()
        if msgs is not None:
            frame = msgs["color"].getCvFrame()
            dets = msgs["detection"].detections

            recog_list = msgs.get("recognition", [])
            eye_list = msgs.get("eye", [])
            if isinstance(eye_list, list):
                if len(eye_list) == 1:
                    eye_scores = np.array(eye_list[0].getFirstLayerFp16()).reshape(-1)
                else:
                    eye_scores = []
            else:
                eye_scores = np.array(eye_list.getFirstLayerFp16()).reshape(-1)

            for i, detection in enumerate(dets):
                xmin, ymin, xmax, ymax = detection.xmin, detection.ymin, detection.xmax, detection.ymax
                bbox = frame_norm(frame, (xmin, ymin, xmax, ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

                # Face recognition
                if i < len(recog_list):
                    features = np.array(recog_list[i].getFirstLayerFp16())
                    conf, name = facerec.new_recognition(features)
                    text.putText(frame, f"{name} {(100*conf):.0f}%", (bbox[0] + 10, bbox[1] + 35))
                else:
                    name = "UNKNOWN"
                    text.putText(frame, name, (bbox[0] + 10, bbox[1] + 35))

                # Eye state
                if eye_scores is None or len(eye_scores) != len(EYE_STATE_CLASSES):
                    text.putText(frame, "Eye: ???", (bbox[0] + 10, bbox[1] + 60))
                else:
                    eye_idx = int(np.argmax(eye_scores))
                    eye_state = EYE_STATE_CLASSES[eye_idx]
                    # eye_conf = eye_scores[eye_idx]
                    # text.putText(frame, f"Eye: {eye_state} ({eye_conf:.2f})", (bbox[0] + 10, bbox[1] + 60))
                    text.putText(frame, f"Eye: {eye_state}", (bbox[0]+10, bbox[1]+60))
                    if name != "UNKNOWN" and eye_state == "open-eyes":
                        notify_parent(name)

            cv2.imshow("color", cv2.resize(frame, (800, 800)))
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
