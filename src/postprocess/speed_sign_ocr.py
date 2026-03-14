import cv2
import pytesseract

def read_speed_number(img_path):

    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 8 -c tessedit_char_whitelist=0123456789"
    )

    text = text.strip()

    if text.isdigit():
        return text

    return None