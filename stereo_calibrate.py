import os

import cv2 as cv
import numpy as np

import sys
import getopt
from glob import glob


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


KONST = 0


def main():
    pattern_size = (9, 6)

    args, img_names = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'threads='])

    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1.0)
    args.setdefault('--threads', 4)

    square_size = float(args.get('--square_size'))
    debug_dir = args.get("--debug")
    threads_num = int(args.get('--threads'))

    imgs_left = sorted(glob(img_names[0]))
    imgs_right = sorted(glob(img_names[1]))

    h, w = cv.imread(imgs_left[0], cv.IMREAD_GRAYSCALE).shape[:2]

    imgp_l, obj_points, rms_l, mtx_l, dstm_l, rvecs_l, tvecs_l = \
        calibrate_single_camera(imgs_left,
                                pattern_size,
                                square_size,
                                debug_dir=debug_dir,
                                threads_num=threads_num
                                )
    imgp_r, obj_points, rms_r, mtx_r, dstm_r, rvecs_r, tvecs_r = \
        calibrate_single_camera(imgs_right,
                                pattern_size,
                                square_size,
                                debug_dir=debug_dir,
                                threads_num=threads_num
                                )

    E, F, R, T, retval = stereo_calibrate(dstm_l, dstm_r, h, imgp_l, imgp_r, mtx_l, mtx_r,
                                          obj_points, w)

    print("\nretval:", retval)
    print("R:\n", R)
    print("T:\n", T)
    print("E:\n", E)
    print("F:\n", F)

    newImageSize = (int(w*1.3), int(h*1.3))
    R_l, R_r, P_l, P_r, Q, roi1, roi2 = \
        cv.stereoRectify(mtx_l, dstm_l,
                         mtx_r, dstm_l,
                         (w, h), R, T,
                         flags=cv.CALIB_ZERO_DISPARITY,
                         alpha=0,
                         newImageSize=newImageSize
                         )
    print("R_l:\n", R_l)
    print("R_r:\n", R_r)
    print("P_l:\n", P_l)
    print("P_r:\n", P_r)
    print("Q:\n", Q)
    np.savetxt("q.np",Q)
    print("roi1:\n", roi1)
    print("roi2:\n", roi2)

    mapx_l, mapy_l = cv.initUndistortRectifyMap(mtx_l, dstm_l, R_l, P_l, newImageSize, cv.CV_16SC2)
    mapx_r, mapy_r = cv.initUndistortRectifyMap(mtx_r, dstm_r, R_r, P_r, newImageSize, cv.CV_16SC2)

    print("mapx_l:\n", mapx_l)
    print("mapy_l:\n", mapy_l)
    print("mapx_r:\n", mapx_r)
    print("mapy_r:\n", mapy_r)

    for file_name_l, file_name_r in zip(imgs_left,imgs_right):
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx_l, dstm_l, (w, h), 1, (w, h))
        undistort_and_write(debug_dir, file_name_l, mapx_l, mapy_l, newcameramtx, mtx_l, dstm_l)
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx_r, dstm_r, (w, h), 1, (w, h))
        undistort_and_write(debug_dir, file_name_r, mapx_r, mapy_r, newcameramtx, mtx_r, dstm_r)

        cv.waitKey(0)



def undistort_and_write(debug_dir, file_name, mapx, mapy, newcameramtx, mtx, dst_coeff):
    frame = cv.imread(file_name)
    stereo_undist = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
    single_undist = cv.undistort(frame, mtx, dst_coeff, None, newcameramtx)

    if debug_dir:
        _path, name, _ext = splitfn(file_name)
        outfile_st_undist = os.path.join(debug_dir, name + '_st_und.png')
        cv.imwrite(outfile_st_undist, stereo_undist)
        cv.imshow(name[:-2],stereo_undist)

        outfile_single_undist = os.path.join(debug_dir, name + '_single_und.png')
        cv.imwrite(outfile_single_undist, single_undist)


def stereo_calibrate(dstm_l, dstm_r, h, imgp_l, imgp_r, mtx_l, mtx_r, obj_points, w):
    flags = cv.CALIB_FIX_INTRINSIC
    T = np.zeros((3, 1), dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv.stereoCalibrate(
        obj_points,
        imgp_l,
        imgp_r,
        mtx_l,
        dstm_l,
        mtx_r,
        dstm_r,
        (w, h),
        R,
        T,
        flags=flags
    )
    return E, F, R, T, retval


def calibrate_single_camera(img_names, pattern_size, square_size, debug_dir=None, threads_num=1):
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[
           :2]  # TODO: use imquery call to retrieve results

    def processImage(file_name):
        print('processing %s... ' % file_name)
        img = cv.imread(file_name, 0)
        if img is None:
            print("Failed to load", file_name)
            return None

        assert w == img.shape[1] and h == img.shape[0], (
                "size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, pattern_size, corners, found)
            _path, name, _ext = splitfn(file_name)
            outfile = os.path.join(debug_dir, name + '_chess.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('chessboard not found')
            return None

        print('           %s... OK' % file_name)
        return (corners.reshape(-1, 2), pattern_points)

    if threads_num <= 1:
        chessboards = [processImage(fn) for fn in img_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, img_names)
    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    rms, mtx_l, dstc_l, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points,
                                                          (w, h), None, None)
    return img_points, obj_points, rms, mtx_l, dstc_l, rvecs, tvecs


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
