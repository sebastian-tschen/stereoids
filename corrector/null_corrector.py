from corrector import ImgCorrector


class NullCorrector(ImgCorrector):
    def get_corrected_generator(self, img_pair_generator):
        return img_pair_generator
