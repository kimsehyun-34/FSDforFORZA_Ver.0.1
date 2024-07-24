# import keyboard
# import numpy as np
# from PIL import ImageGrab
# import cv2
# import time

# 일반화면
# last_time = time.time()
# while(True):
#     # 800x600 windowed mode
#     printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
#     cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
#     print('loop took {} seconds'.format(time.time()-last_time))
#     last_time = time.time()
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

# 반전
# while(True):
#     printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
#     cv2.imshow('window',printscreen)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

import numpy as np
from PIL import ImageGrab
import cv2
import time

def region_of_interest(canny):
    height, width = canny.shape
    mask = np.zeros_like(canny)

    # only focus bottom half of the screen

    polygon = np.array([[
        (0, height*(1/2)),
        (width, height*(1/2)),
        (width, height),
        (0, height),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    new_screen("mask", mask)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    return processed_img

last_time = time.time()
while True:
    screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    #print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    #new_screen = process_img(screen)
    
    l_screen=cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(l_screen)
    
    v = cv2.add(v, -50)
    v = np.clip(v, 0, 255)  # 명도 값이 0-255 범위를 벗어나지 않도록 클리핑
    
    # 채도(S) 채널 조정
    s = cv2.add(s, -255)
    s = np.clip(s, 0, 255)  # 채도 값이 0-255 범위를 벗어나지 않도록 클리핑
    
    # 조정된 채널 병합
    hsv_adjusted = cv2.merge([h, s, v])
    
    # HSV 이미지를 다시 BGR 색 공간으로 변환
    new_screen = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    cv2.imshow('window', new_screen)
    #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break