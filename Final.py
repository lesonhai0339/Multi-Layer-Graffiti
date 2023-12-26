import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tkinter as tk
from tkinter import Tk, filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image
import cv2
from PIL import Image, ImageDraw


def docanh():
    # Tạo một cửa sổ Tkinter
    root = tk.Tk()
    root.withdraw()

    # Hiển thị hộp thoại chọn tệp và lấy đường dẫn tệp được chọn
    file_path = filedialog.askopenfilename()

    # Kiểm tra xem người dùng đã chọn một tệp ảnh hay chưa
    if file_path.endswith('.jpg') or file_path.endswith('.png'):

        # Đọc tệp ảnh bằng OpenCV
        img = cv2.imread(file_path)
    return img
def open():
    root = tk.Tk()
    root.withdraw()

    # Hiển thị hộp thoại chọn tệp và lấy đường dẫn tệp được chọn
    file_path = filedialog.askdirectory()
    return file_path
def save():
    root = tk.Tk()
    root.withdraw()

    # Hiển thị hộp thoại chọn tệp và lấy đường dẫn tệp được chọn
    file_path = filedialog.askdirectory()
    return file_path
def inAnh2():
    img = docanh()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=thaydoivien(img)

    img_flat = img.reshape((-1, 3))
    num_colors = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
    _, labels, centers = cv2.kmeans(img_flat.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Tạo thư mục để lưu ảnh đầu ra #nếu nó chưa tồn tại
    # output_dir = 'output'
    output_dir = save()
    #os.makedirs(output_dir, exist_ok=True)

    # Chuyển các điểm ảnh về màu trung bình của nhóm và lưu từng ảnh
    for i in range(num_colors):
        mask = (labels == i).reshape(img.shape[:2])
        masked_img = img.copy()
        masked_img[~mask] = [0, 0, 0]
        output_path = os.path.join(output_dir , f'output_{i}.jpg')
        cv2.imwrite(output_path, masked_img)
def inAnh():
    # Đọc ảnh
    # img = cv2.imread('santa.png')
    img=docanh()

    # Chuyển đổi sang định dạng RGB nếu cần thiết
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Chuyển ảnh thành một mảng numpy
    img_array = np.array(img)

    # Lấy danh sách các giá trị duy nhất trong mảng 2 chiều của ảnh
    unique_values = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0)

    # Với mỗi giá trị duy nhất, tính toán số lượng điểm ảnh có giá trị đó
    # for value in unique_values:
    #     count = np.count_nonzero(np.all(img_array == value, axis=2))
    #     print(f'{value}: {count} pixels')
    # for color in unique_values:
    #     count = np.count_nonzero(np.all(img_array == color, axis=2))
    #     print(count)
    #     if count >100:
    #         mask = (img == color).all(axis=2)
    #         masked_img = img.copy()
    #         masked_img[~mask] = [0, 0, 0]
    #         masked_img[mask] = color
    #         # plt.imshow(masked_img)
    #         # plt.show()
    #         cv2.imwrite(f'output/{color}.jpg', masked_img)

def ghepAnh():
    # img_folder = "D:/Thunghiem/Test/output/"
    img_folder = open()
    img_files = os.listdir(img_folder)

    images = []
    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        img = cv2.imread(img_path)
        images.append(img)

    canvas_height, canvas_width, _ = images[0].shape
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for image in images:
        canvas = cv2.addWeighted(canvas, 1, image, 1, 0)

    cv2.imshow("Ghep anh", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save_folderpath=save()
    cv2.imwrite(save_folderpath+'/result.png', canvas)

    # # Chuyển đổi ảnh sang không gian màu BGR
    # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # # Duyệt qua từng điểm ảnh trong ảnh
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         pixel_color = image[i, j]

    #         # Kiểm tra xem màu sắc của điểm ảnh có trong danh sách màu cần thay đổi không
    #         if np.all(pixel_color in colors_to_replace):
    #             # Kiểm tra các điểm ảnh láng giềng
    #             neighbors = [
    #                 image[i - 1, j], image[i + 1, j], image[i, j - 1], image[i, j + 1]
    #             ]
    #             neighbor_colors = [np.all(neighbor in colors_to_replace) for neighbor in neighbors]

    #             # Nếu ít nhất một láng giềng có màu không thuộc danh sách, thay đổi màu của điểm ảnh
    #             if any(neighbor_colors):
    #                 image_bgr[i, j] = replacement_color

    # # Chuyển đổi lại ảnh sang không gian màu RGB
    # result_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # return result_image

def ghepAnhxam():
    img_folder = "D:/Thunghiem/Test/output/"
    img_files = os.listdir(img_folder)

    images = []
    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        images.append(img)

    canvas_height, canvas_width, _ = images[0].shape
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for image in images:
        canvas = cv2.addWeighted(canvas, 1, image, 1, 0)

    cv2.imshow("Ghep anh", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output/FinalPicture.jpg', canvas)

def thaydoivien(img):

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    # Phát hiện biên cạnh bằng thuật toán Canny
    edges = cv2.Canny(image, threshold1=30, threshold2=100)
    #Xác định các điểm ảnh viền
    border_pixels = []
    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if edges[i, j] > 0:
                # Kiểm tra các điểm xung quanh để xem xét màu
                neighbor_colors = [img[i-1][j-1] ,img[i - 1, j],img[i-1][j+1],
                                   img[i][j-1],img[i][j+1],
                                   img[i + 1, j-1], img[i+1, j], img[i+1, j + 1]]
                # if np.any([color == [255,255,255] for color in neighbor_colors]):  # Kiểm tra màu xám (128)
                #     border_pixels.append((i, j))
                if np.any(edges[i][j]==255):
                    r_sum, g_sum, b_sum = 0, 0, 0
                    num_neighbors = 0
                    # Lặp qua các pixel kề
                    for neighbor_color in neighbor_colors:
                        r_sum += neighbor_color[0]
                        g_sum += neighbor_color[1]
                        b_sum += neighbor_color[2]
                        num_neighbors += 1

                    # Tính giá trị trung bình của thành phần màu R, G, B
                    if num_neighbors > 0:
                        new_r = r_sum // num_neighbors
                        new_g = g_sum // num_neighbors
                        new_b = b_sum // num_neighbors
                        new_color = [new_r, new_g, new_b]
                    img[i][j]= new_color

    # Chuyển màu của các điểm ảnh viền thành màu xanh (ví dụ)
    # for i, j in border_pixels:
    #     img[i, j] = [255,0,0] # Màu xanh (255)
    # # Lưu ảnh kết quả
    #cv2.imwrite('output1.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

inAnh2() #In hinh anh thanh nhieu hinh con voi cac mau khac nhau
#thaydoivien()
#ghepAnhxam()
#ghepAnh() #ghep hinh anh lai
# docanh()
# Đọc ảnh
# thaydoivien('il_794xN.3691168125_p0zg.jpg')











