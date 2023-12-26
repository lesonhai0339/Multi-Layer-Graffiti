import cv2
import numpy as np
import os

# Đọc ảnh vào trong Python
img = cv2.imread('hinh-anh-chibi-1.jpg')

# Chuyển đổi không gian màu từ RGB sang HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Tạo các mặt nạ cho từng màu
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
red_mask = cv2.inRange(hsv_img, lower_red, upper_red)

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])
green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

# Áp dụng mặt nạ lên ảnh ban đầu để tách các màu
red_img = cv2.bitwise_and(img, img, mask=red_mask)
yellow_img = cv2.bitwise_and(img, img, mask=yellow_mask)
green_img = cv2.bitwise_and(img, img, mask=green_mask)

# Lưu các ảnh tách được vào thư mục
if not os.path.exists('output'):
    os.mkdir('output')
cv2.imwrite('output/red.jpg', red_img)
cv2.imwrite('output/yellow.jpg', yellow_img)
cv2.imwrite('output/green.jpg', green_img)
