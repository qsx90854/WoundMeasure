from model import YOLOV9Seg

from utils.draw_utils import MaskDrawer


class WoundDetector:
    def __init__(self):
        self.model = YOLOV9Seg()
        self.drawer = MaskDrawer()
    
    def predict(self, input_image, draw_result=True):
        output = self.model.predict(input_image)
        if draw_result:
            img = self.drawer(input_image, output[0], idx2label=['wound'])
            return img
        else:
            return output