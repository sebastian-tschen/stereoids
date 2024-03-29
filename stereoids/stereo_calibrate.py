import logging
import os

import cv2 as cv
import numpy as np

import sys
import getopt
from glob import glob

from stereoids import mkdir_p


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


class Calibrator:
    def __init__(self, pattern_size=(7, 6), square_size=1.0, debug_dir=None):
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.size = None

        self.pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        self.pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        self.pattern_points *= self.square_size

        self.debug_dir = debug_dir
        self.log = logging.getLogger(self.__class__.__name__)

    @property
    def h(self):
        return self.size[0]

    @property
    def w(self):
        return self.size[1]

    @property
    def size_wh(self):
        return self.size[::-1]

    def print_camera_values(self, mtx_l, postfix=""):
        fovx, fovy, focalLength, principalPoint, aspectRatio = \
            cv.calibrationMatrixValues(mtx_l, self.size_wh, self.w, self.h)

        self.log.debug("fov_x{}: {}\nfov_y{}: {}".format(postfix, fovx, postfix, fovy))

    def _calibrate_single_camera(self, chessboards):

        obj_points = []
        img_points = []

        for (corners, pattern_points) in [x for x in chessboards if x is not None]:
            img_points.append(corners)
            obj_points.append(pattern_points)

        rms, mtx_l, dstc_l, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points,
                                                              self.size_wh, None, None)
        return img_points, obj_points, rms, mtx_l, dstc_l, rvecs, tvecs

    def _processImage(self, img, image_id=None):
        if img is None:
            return None

        assert self.w == img.shape[1] and self.h == img.shape[0], (
                "size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv.findChessboardCorners(img, self.pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if self.debug_dir and image_id:
            mkdir_p(self.debug_dir)
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, self.pattern_size, corners, found)
            outfile = os.path.join(self.debug_dir, image_id + '_chess.png')
            cv.imwrite(outfile, vis)

        if not found:
            self.log.debug('chessboard not found {}'.format(image_id))
            return None

        return (corners.reshape(-1, 2), self.pattern_points)

    def _stereo_calibrate(self, dstm_l, dstm_r, mtx_l, mtx_r, chessboards):
        flags = cv.CALIB_FIX_INTRINSIC
        T = np.zeros((3, 1), dtype=np.float64)
        R = np.eye(3, dtype=np.float64)

        chessboards_l = [ch_l for ch_l, ch_r in chessboards]
        chessboards_r = [ch_r for ch_l, ch_r in chessboards]

        imgp_l = [corners for corners, pattern_points in chessboards_l]
        obj_points = [pattern_points for corners, pattern_points in chessboards_l]
        imgp_r = [corners for corners, pattern_points in chessboards_r]

        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv.stereoCalibrate(
            obj_points,
            imgp_l,
            imgp_r,
            mtx_l,
            dstm_l,
            mtx_r,
            dstm_r,
            self.size_wh,
            R,
            T,
            flags=flags
        )
        return E, F, R, T, retval

    def calibrate(self, double_frame_generator):

        count = 0;
        self.size = None
        chessboards = list()
        chessboards_l = list()
        chessboards_r = list()
        for img_left, img_right in double_frame_generator:
            if self.size is None:
                self.size = img_left.shape[:2]

            chssbrd_l = self._processImage(img_left, "l_{:02}".format(count))
            chssbrd_r = self._processImage(img_right, "r_{:02}".format(count))


            count += 1
            if chssbrd_l is not None:
                chessboards_l.append(chssbrd_l)
            if chssbrd_r is not None:
                chessboards_r.append(chssbrd_r)
            if chssbrd_r is not None and chssbrd_l is not None:
                chessboards.append((chssbrd_l, chssbrd_r))

        imgp_l, _, rms_l, mtx_l, dstm_l, rvecs_l, tvecs_l = \
            self._calibrate_single_camera(chessboards_l)
        imgp_r, _, rms_r, mtx_r, dstm_r, rvecs_r, tvecs_r = \
            self._calibrate_single_camera(chessboards_r)

        self.log.debug("rms_l:\n{}".format(rms_l))
        self.log.debug("rms_r:\n{}".format(rms_r))
        self.log.debug("mtx_l:\n{}".format(mtx_l))
        self.log.debug("mtx_r:\n{}".format(mtx_r))
        self.print_camera_values(mtx_l, postfix="_l")
        self.print_camera_values(mtx_r, postfix="_r")

        E, F, R, T, retval = self._stereo_calibrate(
            dstm_l,
            dstm_r,
            mtx_l,
            mtx_r,
            chessboards
        )

        self.log.debug("\nretval: {}".format(retval))
        self.log.debug("R:\n{}".format(R))
        self.log.debug("T:\n{}".format(T))
        self.log.debug("E:\n{}".format(E))
        self.log.debug("F:\n{}".format(F))

        newImageSize = (int(self.w * 1), int(self.h * 1))
        R_l, R_r, P_l, P_r, Q, roi1, roi2 = \
            cv.stereoRectify(mtx_l, dstm_l,
                             mtx_r, dstm_r,
                             self.size_wh, R, T,
                             flags=cv.CALIB_ZERO_DISPARITY,
                             alpha=0,
                             newImageSize=newImageSize
                             )
        self.log.debug("R_l:\n{}".format(R_l))
        self.log.debug("R_r:\n{}".format(R_r))
        self.log.debug("P_l:\n{}".format(P_l))
        self.log.debug("P_r:\n{}".format(P_r))
        self.log.debug("Q:\n{}".format(Q))
        self.log.debug("roi1:\n{}".format(roi1))
        self.log.debug("roi2:\n{}".format(roi2))

        mapx_l, mapy_l = cv.initUndistortRectifyMap(mtx_l, dstm_l, R_l, P_l, newImageSize,
                                                    cv.CV_16SC2)
        mapx_r, mapy_r = cv.initUndistortRectifyMap(mtx_r, dstm_r, R_r, P_r, newImageSize,
                                                    cv.CV_16SC2)

        self.log.debug("mapx_l:\n{}".format(mapx_l))
        self.log.debug("mapy_l:\n{}".format(mapy_l))
        self.log.debug("mapx_r:\n{}".format(mapx_r))
        self.log.debug("mapy_r:\n{}".format(mapy_r))

        return mapx_l, mapy_l, mapx_r, mapy_r, Q


class Rectifier:

    def __init__(self, mapx_l, mapy_l, mapx_r, mapy_r):
        self.mapx_l = mapx_l
        self.mapy_l = mapy_l
        self.mapx_r = mapx_r
        self.mapy_r = mapy_r

    def rectify(self, img_l, img_r):
        stereo_undist_l = cv.remap(img_l, self.mapx_l, self.mapy_l, cv.INTER_LINEAR)
        stereo_undist_r = cv.remap(img_r, self.mapx_r, self.mapy_r, cv.INTER_LINEAR)

        return stereo_undist_l, stereo_undist_r


def file_double_image_generator(imgs_left, imgs_right):
    for fn_l, fn_r in zip(imgs_left, imgs_right):
        img_l = cv.cvtColor(cv.imread(fn_l), cv.COLOR_BGR2GRAY)
        img_r = cv.cvtColor(cv.imread(fn_r), cv.COLOR_BGR2GRAY)
        yield img_l, img_r


def main():
    args, img_names = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'threads='])

    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1)
    args.setdefault('--threads', 4)

    square_size = float(args.get('--square_size'))
    debug_dir = args.get("--debug")
    threads_num = int(args.get('--threads'))

    imgs_left = sorted(glob(img_names[0]))
    imgs_right = sorted(glob(img_names[1]))
    if debug_dir:
        logging.basicConfig(level=logging.DEBUG)

    gen = file_double_image_generator(imgs_left, imgs_right)

    calibrator = Calibrator(pattern_size=(6, 7), square_size=square_size, debug_dir=debug_dir)

    mapx_l, mapy_l, mapx_r, mapy_r, Q = calibrator.calibrate(gen)

    np.savetxt("q.txt", Q)
    np.save("q.npy", Q)
    np.save("mapx_l.npy", mapx_l)
    np.save("mapy_l.npy", mapy_l)
    np.save("mapx_r.npy", mapx_r)
    np.save("mapy_r.npy", mapy_r)

    rectifier = Rectifier(mapx_l, mapy_l, mapx_r, mapy_r)

    showimages = True;
    index = 0
    for img_l, img_r in file_double_image_generator(imgs_left, imgs_right):
        if debug_dir:
            rect_l, rect_r = rectifier.rectify(img_l, img_r)
            img_id_l = "l_{:02}".format(index)
            img_id_r = "r_{:02}".format(index)
            outfile_orig_l = os.path.join(debug_dir, img_id_l + '_orig.png')
            outfile_orig_r = os.path.join(debug_dir, img_id_r + '_orig.png')
            outfile_l = os.path.join(debug_dir, img_id_l + '_rect.png')
            outfile_r = os.path.join(debug_dir, img_id_r + '_rect.png')
            cv.imwrite(outfile_orig_l, img_l)
            cv.imwrite(outfile_orig_r, img_r)
            cv.imwrite(outfile_l, rect_l)
            cv.imwrite(outfile_r, rect_r)
            index += 1

        if showimages:
            cv.imshow("left_orig", img_l)
            cv.imshow("right_orig", img_r)
            cv.imshow("left", rect_l)
            cv.imshow("right", rect_r)
            wait_key = cv.waitKey(0)
            key = wait_key & 0xFF
            if key == ord('q'):
                showimages = False


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
