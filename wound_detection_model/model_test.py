import cv2

from wound_detector import WoundDetector

def main():
    test_image = cv2.imread(r'data\3.jpg')

    wound_detector = WoundDetector()

    output = wound_detector.predict(test_image)

    cv2.imshow('image', output)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()