import os

import cv2

if __name__ == '__main__':
    input_dir = './3.jpeg'
    input_name = os.path.basename(input_dir)
    input_name = os.path.splitext(input_name)[0]
    
    input_image = cv2.imread(input_dir, cv2.IMREAD_COLOR)
    
    h, w, _ = input_image.shape
    
    resize_image = cv2.resize(input_image, (w//2, h//2), interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(('{}.png').format(input_name), resize_image)