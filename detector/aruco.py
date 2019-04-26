import cv2.aruco as aruco

from detector import Detector


class ArucoDetector(Detector):

    def __init__(self, dict=aruco.DICT_ARUCO_ORIGINAL, detection_parameter=None):
        self.parameters = aruco.DetectorParameters_create()
        if detection_parameter:
            for param in detection_parameter:
                setattr(self.parameters,param,detection_parameter[param])

        self.aruco_dict = aruco.Dictionary_get(dict)

    def detect(self, frame):
        corners, ids, _ = aruco.detectMarkers(frame, self.aruco_dict, parameters=self.parameters)

        return ids, corners
