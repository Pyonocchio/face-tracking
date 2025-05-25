# register_face.py
# ArcFace 기반 얼굴 임베딩을 저장하는 코드입니다.
# 사용 예시: python register_face.py --name seohyeon

import os
import argparse
import blobconverter
import depthai as dai
import numpy as np
import cv2

# 등록할 이름을 인자로 받음
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, required=True, help="등록할 사람 이름")
args = parser.parse_args()

# 얼굴 임베딩 저장 폴더
databases = "databases"
os.makedirs(databases, exist_ok=True)

# pipeline 생성
pipeline = dai.Pipeline()

# RGB 카메라 노드
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(112, 112)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

# 얼굴 추출용 ImageManip
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(112, 112)
cam.preview.link(manip.inputImage)

# ArcFace 임베딩 추출용 NN
face_rec_nn = pipeline.create(dai.node.NeuralNetwork)
face_rec_nn.setBlobPath(blobconverter.from_zoo(name="face-recognition-arcface-112x112", zoo_type="depthai", shaves=6))
manip.out.link(face_rec_nn.input)

# 출력
arc_xout = pipeline.create(dai.node.XLinkOut)
arc_xout.setStreamName('recognition')
face_rec_nn.out.link(arc_xout.input)

# 영상 출력용 XLinkOut 추가
xout_cam = pipeline.create(dai.node.XLinkOut)
xout_cam.setStreamName("color")
cam.preview.link(xout_cam.input)



with dai.Device(pipeline) as device:
    queue_rgb = device.getOutputQueue(name="color", maxSize=4, blocking=False)
    queue_rec = device.getOutputQueue(name="recognition")

    print(f"[INFO] 얼굴 캡처 중... '{args.name}' 등록을 위해 'q'를 눌러 종료합니다.")

    embeddings = []

    try:
        while True:
            frame = queue_rgb.get().getCvFrame()
            cv2.imshow("Preview", cv2.resize(frame, (800, 800)))

            if queue_rec.has():
                inference = queue_rec.get()
                embedding = np.array(inference.getFirstLayerFp16())
                embeddings.append(embedding)
                print(f"[INFO] 임베딩 {len(embeddings)}개 수집됨")

            # q 키 입력 인식 (안 되는 경우도 대비)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'q' 키 눌러 종료합니다.")
                break

    except KeyboardInterrupt:
        print("[INFO] Ctrl+C로 종료 신호 감지. 저장 루틴으로 진입합니다.")
    finally:
        # 저장 루틴
        if len(embeddings) == 0:
            print("[WARN] 저장할 임베딩이 없습니다.")
        else:
            try:
                with np.load(f"{databases}/{args.name}.npz") as db:
                    db_ = [db[k] for k in db.files]
            except:
                db_ = []

            db_.extend(embeddings)
            np.savez_compressed(f"{databases}/{args.name}", *db_)
            print(f"[INFO] {args.name} 얼굴 임베딩 {len(embeddings)}개 저장 완료")
