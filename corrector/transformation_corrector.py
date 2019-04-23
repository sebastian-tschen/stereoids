from corrector import ImgCorrector
import cv2 as cv


class TransformationCorrector(ImgCorrector):

    def __init__(self, mapx_l, mapy_l, mapx_r, mapy_r):
        self.mapx_l = mapx_l
        self.mapy_l = mapy_l
        self.mapx_r = mapx_r
        self.mapy_r = mapy_r

    def rectify(self, img_l, img_r):
        stereo_undist_l = cv.remap(img_l, self.mapx_l, self.mapy_l, cv.INTER_LINEAR)
        stereo_undist_r = cv.remap(img_r, self.mapx_r, self.mapy_r, cv.INTER_LINEAR)

        return stereo_undist_l, stereo_undist_r

    def corrected_generator(self, img_pair_generator):
        for img_l, img_r in img_pair_generator:
            yield self.rectify(img_l, img_r)

    def get_corrected_generator(self, img_pair_generator):
        return self.corrected_generator(img_pair_generator)
