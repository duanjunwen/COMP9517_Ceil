import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


# img_0 = IMG('./Sequences/01/t010.tif')
class IMG:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(img_path)  # 原图像
        self.gray = cv2.imread(img_path, 0)  # 灰度图
        self.rgb = self.read_rgb_img()  # 彩图
        self.gray_constant_stretch = self.constant_stretch()  # constant stretch对比度拉伸后的灰度图
        self.img_label, self.contours, self.hierarchy = self.find_cell_and_mark()  # contours是一张图的轮廓
        self.contours_centre = self.get_contour_centers()  # 每个轮廓的质心
        self.colours = self.generate_colors()  # 为每个质心生成颜色
        self.average_cell_size = self.average_contours_size()  # 3-2

        # 质心:轮廓类的键值对
        self.contours_centre_and_contours = {self.contours_centre[i]: self.contours[i] for i in
                                             range(len(self.contours_centre))}
        # 质心:外接圆半径
        self.contours_circle = self.get_minEnclosingCircle()

    def read_rgb_img(self):
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        return img_rgb

    def constant_stretch(self):
        res = (self.gray - np.min(self.gray)) * (
                255. / (np.max(self.gray) - np.min(self.gray)))
        res = res.astype(np.uint8)
        return res

    def find_cell_and_mark(self):
        # 对灰度图对比度拉伸
        img_cs = self.constant_stretch()
        # 按threshold二分(大均法)
        _, mask_otsu = cv2.threshold(img_cs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 腐蚀膨胀
        kernel = np.ones((5, 5), np.uint8)
        erode_dilate = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel)
        # 找细胞边界
        # cv.RETR_TREE查找轮廓后轮廓是从外到内的排列顺序
        img_label, contours, hierarchy = cv2.findContours(erode_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 选择size大于30的边界
        contours_30 = [cell for cell in contours if cv2.contourArea(cell) > 30]
        return img_label, contours_30, hierarchy

    def get_contour_centers(self):
        # contour是一张图片种的所有轮廓
        centers = []
        for i, j in zip(self.contours, range(len(self.contours))):
            M = cv2.moments(i)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centers.append((cX, cY))
        return centers

    def get_minEnclosingCircle(self):
        center_radius = dict()
        for i in range(len(self.contours)):
            # 没有使用外接圆心，而是使用了质心
            (x, y), radius = cv2.minEnclosingCircle(self.contours[i])
            center_radius[self.contours_centre[i]] = int(radius)
        return center_radius

    def generate_colors(self):
        colors = []
        while len(colors) != len(self.contours_centre):
            while True:
                red, green, blue = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                if (red, green, blue) not in colors:
                    colors.append((red, green, blue))
                    break
        return colors

    def average_contours_size(self):
        sum_size = 0
        for contour in self.contours:
            sum_size += cv2.contourArea(contour)
        if len(self.contours):
            return round(sum_size / len(self.contours))
        return 0

    def get_draw_contours(self):
        img_o_rgb = cv2.cvtColor(self.gray_constant_stretch, cv2.COLOR_GRAY2RGB)
        # 换不同颜色就加个for循环
        for i in range(len(self.contours)):
            cv2.drawContours(img_o_rgb, self.contours, i, self.colours[i], 3)
        plt.imshow(img_o_rgb)
