# distance.py
import math
import cv2

def calculate_distance(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

class DistanceGuardian:
    def __init__(self, max_distance=1.0):
        self.max_distance = max_distance

    def parse_detections(self, detections_with_xyz):
        results = []
        for i, d1 in enumerate(detections_with_xyz):
            for d2 in detections_with_xyz[i+1:]:
                dist = calculate_distance(d1['xyz'], d2['xyz'])
                results.append({
                    'distance': dist,
                    'dangerous': dist < self.max_distance,
                    'd1': d1,
                    'd2': d2
                })
        return results

    def visualize(self, frame, results):
        for result in results:
            x1, y1 = result['d1']['center']
            x2, y2 = result['d2']['center']
            color = (0, 0, 255) if result['dangerous'] else (255, 255, 0)
            cv2.line(frame, (x1, y1), (x2, y2), color, 1)
            mid_x, mid_y = (x1 + x2)//2, (y1 + y2)//2
            label = f"{result['distance']:.2f}m"
            cv2.putText(frame, label, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
