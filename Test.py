from PIL import Image
import tkinter as tk
from tkinter import Tk, filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image
import cv2
from PIL import Image, ImageDraw
import cv2
import numpy as np

# Tạo một cửa sổ Tkinter
root = tk.Tk()
root.withdraw()

# Hiển thị hộp thoại chọn tệp và lấy đường dẫn tệp được chọn
file_path = filedialog.askopenfilename()

# Kiểm tra xem người dùng đã chọn một tệp ảnh hay chưa
if file_path.endswith('.jpg') or file_path.endswith('.png'):

  # Đọc tệp ảnh bằng OpenCV
  img = cv2.imread(file_path)

  # Kiểm tra xem tệp ảnh có được đọc thành công hay không
  if img is not None:
    # Thực hiện các thao tác xử lý ảnh ở đây
    pass
  else:
    print("Không thể đọc tệp ảnh")
else:
  print("Vui lòng chọn một tệp ảnh hợp lệ")


# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold image
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on image
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Show image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('picture.png', img)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Convert to grayscale
# gray = img.convert('L')

# # Threshold image
# thresh = gray.point(lambda x: 0 if x < 128 else 255, '1')
# cv2.imshow('Gray Image', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Tạo ma trận biến đổi affine
M = np.float32([[1, 0, 50], [0, 1, -25]])

# Áp dụng phép biến đổi affine
transformed = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# Hiển thị ảnh đã được biến đổi
cv2.imshow('Transformed Image', transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
resized = cv2.resize(img, (800, 600))
# Find contours
# contours = Image.new('1', img.size, 0)
contours = Image.new('1', resized, 0)
draw = ImageDraw.Draw(contours)
draw.rectangle((0, 0, img.size[0], img.size[1]), fill=1)
draw.polygon([(100, 100), (200, 100), (150, 200)], fill=0)
del draw

resized = cv2.resize(img, (800, 600))
# Create layers
layers = []
for i in range(5):
    # layer = Image.new('1', img.size, 0)
    layer = Image.new('1', resized, 0)
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, 0, img.size[0], img.size[1]), fill=1)
    draw.ellipse((50+i*20, 50+i*20, 150+i*20, 150+i*20), fill=0)
    del draw
    layers.append(layer)

# Save layers as images
for i, layer in enumerate(layers):
    layer.save(f'layer{i+1}.png')


