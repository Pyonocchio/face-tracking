# TwoStageHostSeqSync 클래스 (eye queue 포함 동기화 지원)
# DepthAI로부터 수신되는 프레임(color), 탐지 결과(detection), 인식 결과(recognition), 눈 상태 분석 결과(eye)를
# 시퀀스 넘버(sequence number)를 기준으로 정확히 동기화(sync)해주는 클래스입니다.

class TwoStageHostSeqSync:
    def __init__(self, include_eye=False):
        self.msgs = {}
        self.include_eye = include_eye

    def add_msg(self, msg, name):
        seq = str(msg.getSequenceNum())
        if seq not in self.msgs:
            self.msgs[seq] = {}

        if name == "recognition":
            if "recognition" not in self.msgs[seq]:
                self.msgs[seq]["recognition"] = []
            self.msgs[seq]["recognition"].append(msg)

        elif name == "eye" and self.include_eye:
            if "eye" not in self.msgs[seq]:
                self.msgs[seq]["eye"] = []
            self.msgs[seq]["eye"].append(msg)

        elif name in ["color", "detection"]:
            self.msgs[seq][name] = msg
            if name == "detection":
                self.msgs[seq]["len"] = len(msg.detections)

    def get_msgs(self):
        seq_remove = []

        for seq, msgs in self.msgs.items():
            if "color" in msgs and "len" in msgs:
                # 👇 수정된 조건
                if "recognition" in msgs and len(msgs["recognition"]) == msgs["len"]:
                    if self.include_eye and ("eye" not in msgs or len(msgs["eye"]) != msgs["len"]):
                        continue
                    seq_remove.append(seq)
                    for s in seq_remove:
                        del self.msgs[s]
                    return msgs
        return None

