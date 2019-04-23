# stereoids
a simple library for stereo computer vision with (aruco) markers

installation
---

`pip install stereoids`

usage examples
----

###calibration
calibration happens via a set of stereo pair images of a chessboard pattern (see _data_ directory)
the capture module will create such images, the stereo_calibrate module processes them and creates a set 
of calibration files for transformation image transformation and a disparity to depth matrix Q.


####create calibration images


```python
from stereoids.capture import Capture

capture = Capture(0, 1, outdir="calib_images")
capture.start()
```

- assuming two video-capture devices with ids 0 and 1 for left and right camera respectively
- to capture a pair of images press 't'
- to finish capture process press 'q'


####calibrate camera pair

```python

from glob import glob

from stereoids.stereo_calibrate import file_double_image_generator, Calibrator
import numpy as np

imgs_left = sorted(glob("calib_images/left*"))
imgs_right = sorted(glob("calib_images/left*"))

gen = file_double_image_generator(imgs_left, imgs_right)

calibrator = Calibrator(pattern_size=(6, 9), square_size=4.0, debug_dir="debug_images")

mapx_l, mapy_l, mapx_r, mapy_r, Q = calibrator.calibrate(gen)

np.savetxt("q.txt", Q)
np.save("q.npy", Q)
np.save("mapx_l.npy", mapx_l)
np.save("mapy_l.npy", mapy_l)
np.save("mapx_r.npy", mapx_r)
np.save("mapy_r.npy", mapy_r)
```

parameters and return values of `Calibrator`:

- pattern_size: the amount of inner-corners in your chessboard (e.g. a normal chessboard with 8x8 fields has 7x7 inner corners)
- square_size: the size of a singe square in the chessboard pattern. In whatever unit you want your measurements in 3D space to be
- debug_dir: a directory to save a set of images of uncorrected and corrected images
- map\<x | y\>_<l | r>: transformation matrices for left and right camera in x and y direction.
- Q: disparity to depth mapping


###locating markers in space

the marker_locate module is responsible for locating markers in 3d space. It requires at a minimum 
a disparity to depth mapping (Q) and a marker-locator to locate markers in images and map them to 3D space. 
Additionally a `Corrector` is recommended for accurate results.

```python
import cv2 as cv
import numpy as np

from corrector.transformation_corrector import TransformationCorrector
from detector.aruco import ArucoDetector
from stereoids.marker_locate import Locator

Q = np.load("q.npy")
mapx_l = np.load("mapx_l.npy")
mapy_l = np.load("mapy_l.npy")
mapx_r = np.load("mapx_r.npy")
mapy_r = np.load("mapy_r.npy")

detector = ArucoDetector()
corrector = TransformationCorrector(mapx_l, mapy_l, mapx_r, mapy_r)


def img_generator(l, r):
    cap_l = cv.VideoCapture(l)
    cap_r = cv.VideoCapture(r)
    while True:
        ret_l, img_l = cap_l.read()
        ret_r, img_r = cap_r.read()
        img_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        img_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
        yield img_l, img_r


def callback(detections):
    for marker_id in detections:
        print(str(marker_id) + "\n" + str(detections[marker_id]))


locator = Locator(Q, detector, corrector)
locator.locate(img_generator(2, 0), callback)
```


