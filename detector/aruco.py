import cv2.aruco as aruco

from detector import Detector


class ArucoDetector(Detector):

    def __init__(self, dict=aruco.DICT_ARUCO_ORIGINAL):
        self.parameters = aruco.DetectorParameters_create()
        self.aruco_dict = aruco.Dictionary_get(dict)

    def detect(self, frame):
        corners, ids, _ = aruco.detectMarkers(frame, self.aruco_dict, parameters=self.parameters)

        return ids, corners
