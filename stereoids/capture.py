import os

import cv2 as cv


class Capture():

    def __init__(self, dev_l, dev_r, outdir="capture", prefix=""):
        self.dev_l = dev_l
        self.dev_r = dev_r
        self.outdir = outdir
        self.prefix = prefix

    def start(self):

        cap_left = cv.VideoCapture(self.dev_l)
        cap_right = cv.VideoCapture(self.dev_r)

        capture_count = 0
        while 1:
            ret_l, img_l = cap_left.read()
            ret_r, img_r = cap_right.read()

            cv.imshow("left", img_l)
            cv.imshow("right", img_r)

            wait_key = cv.waitKey(1)
            key = wait_key & 0xFF
            if key == ord('q'):
                break
            if key == ord('t'):
                outfile_l = os.path.join(self.outdir, self.prefix + 'left_{:2}.png'.format(capture_count))
                outfile_r = os.path.join(self.outdir, self.prefix + 'right_{:2}.png'.format(capture_count))

                cv.imwrite(outfile_l, img_l)
                cv.imwrite(outfile_r, img_r)
                capture_count += 1


def main():
    capture = Capture(2, 0)
    capture.start()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
