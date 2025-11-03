import os

import easyocr
import pandas as pd
from paddleocr import PaddleOCR

from src.methods.base_ocr import BaseOCR


class ModelEasyOCR(BaseOCR):
    def __init__(self, lang='ru') -> None:
        super().__init__()
        self.model = easyocr.Reader([lang])

        self.model_name = "EasyOCR"

    def run_method(self, image_path):
        result = self.model.readtext(image_path)
        full_text = " ".join(x[1] for x in result) 
        return full_text

