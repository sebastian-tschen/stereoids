from abc import ABC, abstractmethod


class ImgCorrector(ABC):

    @abstractmethod
    def get_corrected_generator(self, img_pair_generator):
        pass
