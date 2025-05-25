# TwoStageHostSeqSync 클래스 (eye queue 포함 동기화 지원)
# DepthAI로부터 수신되는 프레임(color), 탐지 결과(detection), 인식 결과(recognition), 눈 상태 분석 결과(eye)를
# 시퀀스 넘버(sequence number)를 기준으로 정확히 동기화(sync)해주는 클래스입니다.

class TwoStageHostSeqSync:
    def __init__(self, include_eye=False):
        self.msgs = {}  # 시퀀스 번호별 메시지 저장
        self.include_eye = include_eye

    def add_msg(self, msg, name):
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = {}

        # recognition은 다수 결과가 오므로 배열로 관리
        if "recognition" not in self.msgs[seq]:
            self.msgs[seq]["recognition"] = []

        if name == "recognition":
            self.msgs[seq]["recognition"].append(msg)
        elif name in ["color", "detection"]:
            self.msgs[seq][name] = msg
            if name == "detection":
                self.msgs[seq]["len"] = len(msg.detections)
        elif name == "eye" and self.include_eye:
            self.msgs[seq]["eye"] = msg

    def get_msgs(self):
        seq_remove = []

        for seq, msgs in self.msgs.items():
            if "color" in msgs and "len" in msgs:
                if len(msgs["recognition"]) == msgs["len"]:
                    if self.include_eye and "eye" not in msgs:
                        continue

                    # 동기화 완료 → 반환 및 삭제
                    seq_remove.append(seq)
                    for s in seq_remove:
                        del self.msgs[s]
                    return msgs
        return None
