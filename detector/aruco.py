import cv2.aruco as aruco

from detector import Detector


class ArucoDetector(Detector):

    def __init__(self, aruco_dict=aruco.DICT_ARUCO_ORIGINAL):
        self.aruco_dict = aruco_dict

    def detect(self, frame):
        corners, ids, _ = aruco.detectMarkers(
            frame,
            aruco.Dictionary_get(self.aruco_dict),
            parameters=aruco.DetectorParameters_create()
        )

        return ids, corners
