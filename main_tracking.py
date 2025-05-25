# main_tracking.py (YOLOv5 object detection 기반 눈 감음 분석 포함)
# - 지정된 사람 얼굴 추적
# - 정확한 3D 좌표(x, y, z) 추출
# - 거리 시각화 + 눈 감음(open/closed) 분석

import os
import argparse
import blobconverter
import cv2
import depthai as dai
import numpy as np
from MultiMsgSync import TwoStageHostSeqSync
from distance import DistanceGuardian

# === 설정 ===
EYE_STATE_BLOB_PATH = "pyono-results/result/best_openvino_2022.1_6shave.blob"
EYE_CLASSES = ['closed-eyes', 'open-eyes']

parser = argparse.ArgumentParser()
parser.add_argument("-target", "--target", type=str, required=True, help="추적할 사람 이름")
args = parser.parse_args()
TARGET_NAME = args.target

# bounding box 변환
def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# YOLOv5 output 파싱
def parse_yolo_detections(output, conf_thres=0.5):
    detections = []
    num_det = len(output) // 7
    output = np.array(output).reshape((num_det, 7))

    for det in output:
        conf = det[4]
        if conf < conf_thres:
            continue
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        detections.append({
            'confidence': conf,
            'class_id': class_id,
            'label': EYE_CLASSES[class_id] if class_id < len(EYE_CLASSES) else 'unknown'
        })
    return detections

# 텍스트 출력 클래스
class TextHelper:
    def __init__(self):
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.7, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.7, self.color, 2, self.line_type)

# 얼굴 인식 클래스
class FaceRecognition:
    def __init__(self, db_path):
        self.db_dic = {}
        self.labels = []
        for file in os.listdir(db_path):
            if file.endswith(".npz"):
                label = os.path.splitext(file)[0]
                with np.load(os.path.join(db_path, file)) as db:
                    self.db_dic[label] = [db[k] for k in db.files]
                    self.labels.append(label)

    def cosine_distance(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b.T) / (a_norm * b_norm)

    def recognize(self, feature):
        max_sim = -1
        match_label = "UNKNOWN"
        for label in self.labels:
            for db_vec in self.db_dic[label]:
                sim = self.cosine_distance(feature, db_vec)
                if sim > max_sim:
                    max_sim = sim
                    match_label = label
        return match_label, max_sim

# Pipeline 구성
pipeline = dai.Pipeline()

# RGB 카메라
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(300, 300)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

# StereoDepth
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

# 눈 상태 분석
eye_manip = pipeline.create(dai.node.ImageManip)
eye_manip.initialConfig.setResize(416, 416)
eye_manip.setMaxOutputFrameSize(3 * 416 * 416)
cam.preview.link(eye_manip.inputImage)

eye_nn = pipeline.create(dai.node.NeuralNetwork)
eye_nn.setBlobPath(EYE_STATE_BLOB_PATH)
eye_manip.out.link(eye_nn.input)

# 얼굴 탐지
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(300, 300)
cam.preview.link(manip.inputImage)

det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
det_nn.setConfidenceThreshold(0.5)
det_nn.input.setBlocking(False)
manip.out.link(det_nn.input)
stereo.depth.link(det_nn.inputDepth)

# ArcFace
rec_manip = pipeline.create(dai.node.ImageManip)
rec_manip.initialConfig.setResize(112, 112)
cam.preview.link(rec_manip.inputImage)

rec_nn = pipeline.create(dai.node.NeuralNetwork)
rec_nn.setBlobPath(blobconverter.from_zoo(name="face-recognition-arcface-112x112", zoo_type="depthai", shaves=6))
rec_manip.out.link(rec_nn.input)

# 출력 큐
for name, node in zip(["color", "detection", "recognition", "eye"], [cam.preview, det_nn.out, rec_nn.out, eye_nn.out]):
    out = pipeline.create(dai.node.XLinkOut)
    out.setStreamName(name)
    node.link(out.input)

# 실행 루프
with dai.Device(pipeline) as device:
    recog = FaceRecognition("databases")
    text = TextHelper()
    guardian = DistanceGuardian(max_distance=1.5)
    sync = TwoStageHostSeqSync(include_eye=True)  # eye 동기화 포함

    queues = {name: device.getOutputQueue(name) for name in ["color", "detection", "recognition", "eye"]}

    while True:
        for name, q in queues.items():
            if q.has(): sync.add_msg(q.get(), name)
        msgs = sync.get_msgs()
        if msgs is None: continue

        frame = msgs["color"].getCvFrame()
        detections = msgs["detection"].detections
        detections_with_xyz = []

        # 눈 상태 추론
        eye_status = "unknown"
        try:
            eye_output = msgs["eye"].getFirstLayerFp16()
            parsed = parse_yolo_detections(eye_output)
            if parsed:
                eye_status = parsed[0]['label']
        except: pass

        for i, det in enumerate(detections):
            bbox = frame_norm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            x = det.spatialCoordinates.x / 1000
            y = det.spatialCoordinates.y / 1000
            z = det.spatialCoordinates.z / 1000

            feature = np.array(msgs["recognition"][i].getFirstLayerFp16())
            name, conf = recog.recognize(feature)

            text.putText(frame, f"{name} ({conf*100:.0f}%)", (bbox[0], bbox[1]-100))
            text.putText(frame, f"Eyes: {eye_status}", (bbox[0], bbox[1]-80))
            text.putText(frame, f"X: {x:.2f}m", (bbox[0], bbox[1]-60))
            text.putText(frame, f"Y: {y:.2f}m", (bbox[0], bbox[1]-40))
            text.putText(frame, f"Z: {z:.2f}m", (bbox[0], bbox[1]-20))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)

            detections_with_xyz.append({ 'xyz': (x,y,z), 'center': (center_x, center_y), 'name': name })

            if name == TARGET_NAME and conf >= 0.5:
                print(f"[추적] {name} 위치: X={x:.2f} Y={y:.2f} Z={z:.2f}, 눈: {eye_status}")
            else:
                print(f"[무시] {name} 은(는) 추적 대상 아님")

        guardian.visualize(frame, guardian.parse_detections(detections_with_xyz))
        cv2.imshow("Tracking", cv2.resize(frame, (800, 800)))
        if cv2.waitKey(1) == ord('q'): break
