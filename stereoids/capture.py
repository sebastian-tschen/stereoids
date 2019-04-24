import os

import cv2 as cv

from stereoids import mkdir_p


class Capture():

    def __init__(self, dev_l, dev_r, outdir="capture", prefix="", pattern_size=None, index_start=0):
        self.dev_l = dev_l
        self.dev_r = dev_r
        self.outdir = outdir
        self.prefix = prefix
        self.pattern_size = pattern_size
        self.index_start = index_start

    def start(self):

        cap_left = cv.VideoCapture(self.dev_l)
        cap_right = cv.VideoCapture(self.dev_r)

        capture_count = self.index_start
        while 1:
            cap_left.grab()
            cap_right.grab()
            ret_l, img_l = cap_left.retrieve()
            ret_r, img_r = cap_right.retrieve()

            if self.pattern_size:
                img_bw_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
                img_bw_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
                retval, corners = cv.findChessboardCorners(img_bw_l, patternSize=self.pattern_size)
                img_bw_l = cv.cvtColor(img_bw_l, cv.COLOR_GRAY2BGR)
                cv.drawChessboardCorners(img_bw_l, self.pattern_size, corners, retval)
                retval, corners = cv.findChessboardCorners(img_bw_r, patternSize=self.pattern_size)
                img_bw_r = cv.cvtColor(img_bw_r, cv.COLOR_GRAY2BGR)
                cv.drawChessboardCorners(img_bw_r, self.pattern_size, corners, retval)

                cv.imshow("left", img_bw_l)
                cv.imshow("right", img_bw_r)
            else:
                cv.imshow("left", img_l)
                cv.imshow("right", img_r)

            wait_key = cv.waitKey(1)
            key = wait_key & 0xFF
            if key == ord('q'):
                break
            if key == ord('t'):
                outfile_l = os.path.join(self.outdir, self.prefix + 'left_{:02}.png'.format(capture_count))
                outfile_r = os.path.join(self.outdir, self.prefix + 'right_{:02}.png'.format(capture_count))

                mkdir_p(self.outdir)
                cv.imwrite(outfile_l, img_l)
                cv.imwrite(outfile_r, img_r)
                capture_count += 1


def main():
    capture = Capture(0, 2, pattern_size=(6, 7), index_start=21)
    capture.start()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
