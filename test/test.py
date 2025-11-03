from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang='ru',
    text_recognition_model_name="PP-OCRv4_mobile_rec",
    use_doc_orientation_classify=False, # Use use_doc_orientation_classify to enable/disable document orientation classification model
    use_doc_unwarping=False, # Use use_doc_unwarping to enable/disable document unwarping module
    use_textline_orientation=True, # Use use_textline_orientation to enable/disable textline orientation classification model
)
result = ocr.predict("test/4.png")  

print(" ".join(result[0]['rec_texts']))
