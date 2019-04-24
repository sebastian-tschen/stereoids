"""This module will identify markers of a given type and try to position them in 3d space"""
import argparse

import cv2 as cv
import numpy as np

from corrector import ImgCorrector
from corrector.transformation_corrector import TransformationCorrector
from detector import Detector
from detector.aruco import ArucoDetector


def find_unique_markers(marker_ids, corners):
    unique_markers = {}

    seen_ids = set()
    for i, marker_id in enumerate(marker_ids):

        int_marker_id = int(marker_id)
        if int_marker_id in seen_ids:
            try:
                unique_markers.pop(int_marker_id)
            except KeyError:
                pass
        else:
            seen_ids.add(int_marker_id)
            unique_markers[int_marker_id] = corners[i]
    return unique_markers


def find_marker_pairs(ids_l, corners_l, ids_r, corners_r):
    unique_l = find_unique_markers(ids_l, corners_l)
    unique_r = find_unique_markers(ids_r, corners_r)

    marker_pairs = {}
    for id_l in unique_l:
        if id_l in unique_r:
            marker_pairs[id_l] = (unique_l[id_l], unique_r[id_l])

    return marker_pairs


def get_position_from_corners(corners):

    return (corners[0][0] + corners[0][1] + corners[0][2] + corners[0][3]) / 4


def calculate_marker_disparity(marker_pairs):
    disparity_map = {}
    for marker_id in marker_pairs:
        corners_l, corners_r = marker_pairs[marker_id]
        pos_l = get_position_from_corners(corners_l)
        pos_r = get_position_from_corners(corners_r)

        disparity = pos_l[0] - pos_r[0];

        disparity_map[marker_id] = np.array([[[pos_l[0], pos_l[1], disparity]]])
    return disparity_map


class Locator:

    def __init__(self, Q, marker_detector: Detector, img_corrector: ImgCorrector = None):
        self.Q = Q
        self.marker_detector = marker_detector
        self.img_corrector = img_corrector

    def calculate_marker_positions(self, marker_disparity):

        marker_positions = {}
        for marker_id in marker_disparity:
            position = cv.perspectiveTransform(marker_disparity[marker_id], self.Q)
            marker_positions[marker_id] = position

        return marker_positions

    def locate(self, img_generator, callback):

        if self.img_corrector is not None:
            img_generator = self.img_corrector.get_corrected_generator(img_generator)

        for img_l, img_r in img_generator:
            ids_l, corners_l = self.marker_detector.detect(img_l)
            ids_r, corners_r = self.marker_detector.detect(img_r)

            if ids_l is not None and ids_r is not None:
                marker_pairs = find_marker_pairs(ids_l, corners_l, ids_r, corners_r)
                marker_disparity = calculate_marker_disparity(marker_pairs)
                marker_positions = self.calculate_marker_positions(marker_disparity)
                callback(marker_positions)
            else:
                callback({})


def main():
    parser = argparse.ArgumentParser(description='start a test marker location')

    parser.add_argument('--Q', type=str, default="q.npy", help='disparity-to-depth mapping matrix')
    parser.add_argument('--mapx_l', type=str, default="mapx_l.npy", help='transformation map x director left')
    parser.add_argument('--mapy_l', type=str, default="mapy_l.npy", help='transformation map y director left')
    parser.add_argument('--mapx_r', type=str, default="mapx_r.npy", help='transformation map x director right')
    parser.add_argument('--mapy_r', type=str, default="mapy_r.npy", help='transformation map y director right')
    args = parser.parse_args()

    # read correction parameter
    Q = np.load(args.Q)
    mapx_l = np.load(args.mapx_l)
    mapy_l = np.load(args.mapy_l)
    mapx_r = np.load(args.mapx_r)
    mapy_r = np.load(args.mapy_r)

    detector = ArucoDetector()
    corrector = TransformationCorrector(mapx_l, mapy_l, mapx_r, mapy_r)

    locator = Locator(Q, detector, corrector)

    def img_generator(l, r):
        cap_l = cv.VideoCapture(l)
        cap_r = cv.VideoCapture(r)

        while 1:
            cap_l.grab()
            cap_r.grab()
            ret_l, img_l = cap_l.retrieve()
            ret_r, img_r = cap_r.retrieve()

            cv.imshow("left", img_l)
            cv.imshow("right", img_r)

            wait_key = cv.waitKey(1)
            if wait_key != -1:
                break

        cv.destroyAllWindows()
        while True:
            cap_l.grab()
            cap_r.grab()
            ret_l, img_l = cap_l.retrieve()
            ret_r, img_r = cap_r.retrieve()
            img_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
            img_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
            yield img_l, img_r

    def callback(detections):

        for marker_id in detections:
            print(str(marker_id) + "\n" + str(detections[marker_id]))

    locator.locate(img_generator(2, 0), callback)


if __name__ == "__main__":
    main()

    cv.destroyAllWindows()
