import cv2
import numpy as np
from PIL import ImageGrab
from directkeys import ReleaseKey, PressKey, W, A, S, D
import time

def process_frame(image):
    
    # l_screen=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(l_screen)
    
    # v = cv2.add(v, -50)
    # v = np.clip(v, 0, 255)  # 명도 값이 0-255 범위를 벗어나지 않도록 클리핑
    
    # # 채도(S) 채널 조정
    # s = cv2.add(s, -255)
    # s = np.clip(s, 0, 255)  # 채도 값이 0-255 범위를 벗어나지 않도록 클리핑
    
    # # 조정된 채널 병합
    # hsv_adjusted = cv2.merge([h, s, v])
    
    # # HSV 이미지를 다시 BGR 색 공간으로 변환
    # new_screen = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    # Convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # GPU-accelerated edge detection
    canny_detector = cv2.cuda.createCannyEdgeDetector(200, 300)
    gpu_gray = cv2.cuda_GpuMat()
    gpu_gray.upload(processed_img)
    gpu_canny = canny_detector.detect(gpu_gray)
    processed_img = gpu_canny.download()

    # CPU-based Gaussian blur
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    
    # Define region of interest (ROI) - expand the ROI
    height, width = processed_img.shape
    mask = np.zeros_like(processed_img)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(0.9 * width), int(height * 0.5)),
        (int(0.1 * width), int(height * 0.5)),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(processed_img, mask)
    
    # Hough line detection with adjusted parameters
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=200)

    
    return lines

def draw_lines(frame, lines):
    if lines is None or len(lines) == 0:
        return None, None, None
    
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.5:
            left_lines.append(line)
        elif slope > 0.5:
            right_lines.append(line)
    
    height, width, _ = frame.shape
    left_line_coords = None
    right_line_coords = None
    mid_line_coords = None

    if left_lines:
        left_line = np.mean(left_lines, axis=0).astype(int)
        x1, y1, x2, y2 = left_line[0]
        y1_new = height
        y2_new = int(height * 0.5)
        x1_new = int(x1 + (y1_new - y1) * (x2 - x1) / (y2 - y1))
        x2_new = int(x1 + (y2_new - y1) * (x2 - x1) / (y2 - y1))
        cv2.line(frame, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 10)
        left_line_coords = (x1_new, y1_new, x2_new, y2_new)
        
    if right_lines:
        right_line = np.mean(right_lines, axis=0).astype(int)
        x1, y1, x2, y2 = right_line[0]
        y1_new = height
        y2_new = int(height * 0.5)
        x1_new = int(x1 + (y1_new - y1) * (x2 - x1) / (y2 - y1))
        x2_new = int(x1 + (y2_new - y1) * (x2 - x1) / (y2 - y1))
        cv2.line(frame, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 10)
        right_line_coords = (x1_new, y1_new, x2_new, y2_new)
        
    if left_lines and right_lines:
        left_line = np.mean(left_lines, axis=0).astype(int)
        right_line = np.mean(right_lines, axis=0).astype(int)
        mid_x1 = (left_line[0][0] + right_line[0][0]) // 2
        mid_y1 = int(height * 0.7)
        mid_x2 = (left_line[0][2] + right_line[0][2]) // 2
        mid_y2 = int(height * 0.7)
        cv2.line(frame, (mid_x1, mid_y1), (mid_x2, mid_y2), (255, 0, 0), 10)
        mid_line_coords = (mid_x1, mid_y1, mid_x2, mid_y2)

    return left_line_coords, right_line_coords, mid_line_coords

def determine_position(frame, left_line_coords, right_line_coords, mid_line_coords):
    if mid_line_coords is None:
        print('중앙선을 찾을 수 없습니다.')
        return
    
    height, width, _ = frame.shape
    mid_x1, mid_y1, mid_x2, mid_y2 = mid_line_coords
    frame_mid_x = width // 2
    if mid_x1 < frame_mid_x - 80:
        print('왼쪽')
        PressKey(A)
        time.sleep(0.05)
        ReleaseKey(A)
    elif mid_x2 > frame_mid_x + 80:
        print('오른쪽')
        PressKey(D)
        time.sleep(0.05)
        ReleaseKey(D)
    else:
        print('중앙')
        PressKey(W)
        ReleaseKey(W)
            
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)

def slow_ya_roll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

last_time = time.time()
while True:
    screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    new_screen=cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    
    # Simulate game frame capture
    # Here you would capture the game screen frame
    # For example, use pyautogui to capture a screenshot and resize to 800x600
    # frame = cv2.resize(pyautogui.screenshot(), (800, 600))
    
    print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()

    lines = process_frame(new_screen)
    draw_lines(new_screen, lines)

    #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    cv2.imshow('window',new_screen)
    
    left_line_coords, right_line_coords, mid_line_coords = draw_lines(new_screen, lines)
    
    #determine_position(new_screen, left_line_coords, right_line_coords)
    height, width, _ = new_screen.shape
    
    if left_line_coords is None or right_line_coords is None:
            print('차선을 찾을 수 없습니다.')
    else:
        determine_position(new_screen, left_line_coords, right_line_coords, mid_line_coords)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

