import math
import os
import random
from collections import defaultdict
from itertools import cycle
import cv2
import matplotlib.pyplot as plt
from IMG_Class import *


# experiment = Experiment('./Sequences/01/')
class Experiment:
    def __init__(self, img_root_path):
        self.img_root_path = img_root_path
        self.img_paths = [img_root_path + x for x in os.listdir(img_root_path)]
        self.IMG_Classes = [IMG(path) for path in self.img_paths]
        self.Init_IMG = self.IMG_Classes[0]  # 初始化第一张图
        self.colors, self.index = self.generate_colors_index()  # 存放{centre:colors}{centre:index}
        self.dividing_cells_group = []  # 正在分裂的细胞群
        self.curr_divided_cell_center = []  # 存分裂的细胞的中心坐标，用于画圆标记
        self.cell_count = len(self.Init_IMG.contours_centre)  # 3-1
        self.tracking_line_colors = defaultdict(list)  # 存的是{color:[这个color的质心们]}
        self.average_displacement = 0  # 3-3  细胞的平均位移
        self.dividing_cells_number = 0  # 3-4
        self.plt_saver = []  # 存储每一张图片

    def generate_colors_index(self):
        contours_centre = self.Init_IMG.contours_centre  # 根据contours_centre生成color和index
        colors = []
        index = []
        while len(colors) != len(contours_centre):
            while True:
                red, green, blue = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                if (red, green, blue) not in colors:
                    colors.append((red, green, blue))
                    break
        for i in range(len(contours_centre)):
            index.append(i)
        colors_centre = dict(zip(contours_centre, colors))
        index_centre = dict(zip(contours_centre, index))
        return colors_centre, index_centre

    # 输入两个图象类
    # 返回
    def find_centre_in_circle(self, IMG0, IMG1, times_radius=1.7):
        self.curr_divided_cell_center = []  # 初始化当前图分裂细胞
        self.cell_count = len(IMG1.contours_centre)  # 3-1：更新细胞数目
        self.dividing_cells_number = 0  # 3-4

        img0_center_circle = IMG0.contours_circle  # 图象0的轮廓{质心:外接圆半径}字典
        img1_centers = IMG1.contours_centre  # 图象1的轮廓质心list
        img0_in_circle = defaultdict(list)
        for img0_center, radius in img0_center_circle.items():
            for img1_center in img1_centers:
                distance = math.sqrt((img1_center[0] - img0_center[0]) ** 2 + (img1_center[1] - img0_center[1]) ** 2)
                if distance < times_radius * radius:
                    img0_in_circle[img0_center].append((img1_center, distance))
            # 如果外接圆内质心数量超过三个，选其中距离最小的两个
            if len(img0_in_circle[img0_center]) > 2:
                img0_in_circle[img0_center] = sorted(img0_in_circle[img0_center], key=lambda x: x[1])[:2]

            # 如果分裂成了两个，更新self.curr_divided_cell_center,用于标记分裂细胞
            if len(img0_in_circle[img0_center]) == 2:
                # 求中心
                # print(img0_in_circle[img0_center])
                (center_x, center_y) = (img0_in_circle[img0_center][0][0][0] + img0_in_circle[img0_center][1][0][
                    0]) // 2, (
                                               img0_in_circle[img0_center][0][0][1] + img0_in_circle[img0_center][1][0][
                                           1]) // 2
                self.dividing_cells_number += 1
                # 如果不重复
                if (int(center_x), int(center_y)) not in self.curr_divided_cell_center:
                    self.curr_divided_cell_center.append((int(center_x), int(center_y)))

        return img0_in_circle

    # 生成一个新color
    # generate_new_color() --》 (0,255,255)
    def generate_new_color(self):
        while True:
            red, green, blue = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            if (red, green, blue) not in self.colors.values():
                return (red, green, blue)

    # 生成一个新index
    # generate_new_index()  --> 25
    def generate_new_index(self):
        return max(self.index.values()) + 1

    # 输入(x0,y0), (x1,y1)，计算两点欧氏距离
    def Euclidean_Distance(self, center_a, center_b):
        return math.sqrt((center_b[0] - center_a[0]) ** 2 + (center_b[1] - center_a[1]) ** 2)

    # 输入前一张图片和后一张图片，更新self.color
    def update_colors(self, IMG0, IMG1):
        moving_cell = 0  # 3-3,记录移动细胞个数
        moving_distance = 0  # 3-3,记录移动细胞长度
        res = self.find_centre_in_circle(IMG0, IMG1)
        center1_remain = set(IMG1.contours_centre)  #
        for center0, center1 in res.items():
            # print(center0, center1)
            if center0 in self.colors.keys():
                # 只有1个center1与之对应,color继承
                if len(center1) == 1:
                    self.colors[center1[0][0]] = self.colors[center0]

                    # 更新tracking color的追踪先
                    if (center1[0][0], center0) not in self.tracking_line_colors[self.colors[center0]] and (
                            center0, center1[0][0]) not in self.tracking_line_colors[self.colors[center0]]:
                        self.tracking_line_colors[self.colors[center0]].append((center1[0][0], center0))
                        moving_cell += 1
                        moving_distance += self.Euclidean_Distance(center0, center1[0][0])

                    self.colors.pop(center0)
                    center1_remain.discard(center1[0][0])
                # 有2个center1与之对应，生成new color
                elif len(center1) == 2:
                    new_color1 = self.generate_new_color()
                    self.colors[center1[0][0]] = new_color1

                    new_color2 = self.generate_new_color()
                    self.colors[center1[1][0]] = new_color2

                    self.colors.pop(center0)
                    center1_remain.discard(center1[0][0])
                    center1_remain.discard(center1[1][0])
        # 还有一部分新cell遗留在center1_remain里
        while center1_remain:
            center1 = center1_remain.pop()
            new_color = self.generate_new_color()
            self.colors[center1] = new_color

        # 删除在img1中消失的cell
        disappear_center = []
        for center, color in self.colors.items():
            if center not in IMG1.contours_centre:
                disappear_center.append(center)
        # 对于消失的cell，删除他所拥有的颜色
        while disappear_center:
            del_center = disappear_center.pop()
            self.colors.pop(del_center)
        self.average_displacement = int(moving_distance // moving_cell)  # 3-3更新平均移动距离

    # 输入前一张图片和后一张图片，更新self.index
    def update_index(self, IMG0, IMG1):
        res = self.find_centre_in_circle(IMG0, IMG1)
        center1_remain = set(IMG1.contours_centre)  #
        for center0, center1 in res.items():
            # print(center0, center1)
            if center0 in self.index.keys():
                # 只有1个center1与之对应,index继承
                if len(center1) == 1:
                    self.index[center1[0][0]] = self.index[center0]
                    self.index.pop(center0)
                    center1_remain.discard(center1[0][0])
                # 有2个center1与之对应，生成new index
                elif len(center1) == 2:
                    new_index1 = self.generate_new_index()
                    self.index[center1[0][0]] = new_index1

                    new_index2 = self.generate_new_index()
                    self.index[center1[1][0]] = new_index2

                    self.index.pop(center0)
                    center1_remain.discard(center1[0][0])
                    center1_remain.discard(center1[1][0])
        # 还有一部分新cell遗留在center1_remain里
        while center1_remain:
            center1 = center1_remain.pop()
            new_index = self.generate_new_index()
            self.index[center1] = new_index

        # 删除在img1中消失的cell
        disappear_center = []
        for center, index in self.index.items():
            if center not in IMG1.contours_centre:
                disappear_center.append(center)
        while disappear_center:
            del_center = disappear_center.pop()
            self.index.pop(del_center)

    ##################################################################################
    # 以下为各式各样的输出图片函数

    # 根据当前self.colors找出IMG对应contours，打印
    def plot_curr_img(self, IMG1):
        img_o_rgb = cv2.cvtColor(IMG1.gray_constant_stretch, cv2.COLOR_GRAY2RGB)
        # 遍历更新过后的self.colors， 存放的是{centre:colors}
        # 生成 contours和colors两个list，内部一一对应
        # 根据centre找contours
        contours = []
        colors = []
        for center, color in self.colors.items():
            if center in IMG1.contours_centre_and_contours.keys():
                contours.append(IMG1.contours_centre_and_contours[center])
                colors.append(color)
        # 添加轮廓
        for i in range(len(contours)):
            cv2.drawContours(img_o_rgb, contours, i, colors[i], 3)

        # 添加index
        for center, index in self.index.items():
            cv2.putText(img_o_rgb, str(index), center, 1, 1, (0, 255, 0), 3)

        # 用红圈标记分裂细胞
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        # 参数：
        # image:它是要在其上绘制圆的图像。
        # center_coordinates：它是圆的中心坐标。坐标表示为两个值的元组，即(X坐标值，Y坐标值)。
        # radius:它是圆的半径。
        # color:它是要绘制的圆的边界线的颜色。对于BGR，我们通过一个元组。例如：(255，0，0)为蓝色。
        # thickness:它是圆边界线的粗细像素。厚度-1像素将以指定的颜色填充矩形形状。
        for divided_center in self.curr_divided_cell_center:
            cv2.circle(img_o_rgb, divided_center, 30, (255, 0, 0), 3)

        # 3-3
        # 用和细胞相同颜色的线标记细胞移动
        #  cv2.line(image, start_point, end_point, color, thickness)
        # cv2.line(img_o_rgb, (391, 676), (392, 675), (251, 212, 26), thickness)
        for tracking_color, center_list in self.tracking_line_colors.items():
            for start_center, end_center in center_list:
                cv2.line(img_o_rgb, start_center, end_center, tracking_color, 3)

        self.plt_saver.append(img_o_rgb)
        plt.imshow(img_o_rgb)

    def write_all_img(self):
        write_path = self.img_root_path.replace('Sequences', 'Result')
        # print(write_path)
        for i in range(len(self.plt_saver)):
            cv2.imwrite(f'{write_path}{i}.png', self.plt_saver[i])

    # 在figure中打印所有图片
    def plt_all_img_in_figure(self):
        fig = plt.figure(figsize=(10, 10))
        title_list = [str(i) for i in range(len(self.plt_saver))]
        for i in range(len(self.plt_saver)):
            ax1 = fig.add_subplot(10, 10, i + 1)
            ax1.imshow(self.plt_saver[i])
            ax1.set_title(title_list[i])
            ax1.set_axis_off()

    # 对指定文件夹下所有
    def plot_one_by_one(self):
        read_path = self.img_root_path.replace('Sequences', 'Result')
        filenames = os.listdir(read_path)
        img_iter = cycle([cv2.imread(os.sep.join([read_path, x])) for x in filenames])
        key = 0
        count = 0
        while key & 0xFF != 27:
            cv2.imshow(f"{count}", next(img_iter))
            count += 1
            key = cv2.waitKey(1000)

    # 批量运行一次实验，循环调用plot_curr_img
    def run_whole_experiment(self):
        write_path = self.img_root_path.replace('Sequences', 'Result')
        for i in range(1, len(self.IMG_Classes)):
            self.update_colors(self.IMG_Classes[i - 1], self.IMG_Classes[i])
            self.update_index(self.IMG_Classes[i - 1], self.IMG_Classes[i])
            self.plot_curr_img(self.IMG_Classes[i])
            print(f'Compared images are {i - 1} and {i}.',)
            print('3-1:cell count ', self.cell_count)
            print('3-2:The average size (in pixels) of all the cells ', self.IMG_Classes[i].average_cell_size)
            print('3-3:The average displacement (in pixels) of all the cells ', self.average_displacement)
            print('3-4:The number of cells that are in the process of dividing', self.dividing_cells_number)
            curr_img = cv2.cvtColor(self.plt_saver[-1], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{write_path}{i}.png', curr_img)
            print()
