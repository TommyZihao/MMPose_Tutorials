from mmdeploy_runtime import Detector
from mmdeploy_runtime import PoseDetector
import cv2
import argparse
import numpy as np
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Pose detection demo.')
    parser.add_argument('detect', help='detection model')
    parser.add_argument('pose', help='pose model')
    parser.add_argument('--file', type=str, help='image/video file')
    parser.add_argument('--camera', type=int, default=0, help='camera id')
    parser.add_argument('--device', default='cpu', help='cpu/cuda device')
    args = parser.parse_args()
    return args


def get_skeleton():
    # (color)
    kep_point = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # (u, v, color)
    skeleton = [(0, 1, (100, 150, 200)), (1, 2, (200, 100, 150)),
                (2, 0, (150, 120, 100))]
    return kep_point, skeleton


def main():
    args = parse_args()
    if args.file:
        cap = cv2.VideoCapture(args.file)
    else:
        cap = cv2.VideoCapture(args.camera)

    bbox_detector = Detector(args.detect, args.device)
    pose_detector = PoseDetector(args.pose, args.device)

    kp, skeleton = get_skeleton()

    while True:
        flag, frame = cap.read()
        if not flag:
            break

        bboxes, labels, masks = bbox_detector(frame)
        # remove low score bbox
        bboxes = bboxes[bboxes[:, -1] > 0.3]
        # remove score
        bboxes = bboxes[:, :4].astype(np.int32)
        # n_box x n_key_point x 3
        key_points = pose_detector(frame, bboxes)

        # draw bbox
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4].astype(np.int32)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # draw key point and skeleton
        for item in key_points:
            # skeleton
            for skt in skeleton:
                p1 = item[skt[0], :2].astype(np.int32)
                p2 = item[skt[1], :2].astype(np.int32)
                frame = cv2.line(frame, p1, p2, skt[2], 3)
            # key point
            for index, p in enumerate(item):
                frame = cv2.circle(frame, p[:2].astype(
                    np.int32), 5, kp[index], cv2.FILLED)

        cv2.imshow('show', frame)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()

