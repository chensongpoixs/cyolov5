import cv2
import imutils
import numpy as np
import pytesseract
import cv2
import pytesseract
from PIL import Image
import cv2
import shutil
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = 'D:\\Tesseract-OCR\\tesseract.exe'

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]


if __name__ == '__main__':

    path = 'D:/Work/cartificial_intelligence/datasets/CCPD2020/ccpd_green/test/result_images/';
    for filename in os.listdir(path):
        new_file_name_path = path + filename;
        list_plate = filename.split('_');
        new_file_name = provinces[int(list_plate[0])] + list_plate[1];
        # print(new_file_name);
        img = cv2.imread(new_file_name_path );
        """OCR识别"""
        text = pytesseract.image_to_string(img, config='--psm 11');
        # text = pytesseract.image_to_string(img,   lang='chi_sim');#psm 12
        print("[new_file_name = " ,new_file_name, "]车牌结果：", text);
        cv2.destroyAllWindows();


    # img = cv2.imread('D:/Work/cartificial_intelligence/datasets/CCPD2020/ccpd_green/test/result_images/0_AD0704.jpg');
    # img = cv2.imread('D:/Work/cartificial_intelligence/datasets/VOC/test.jpg')
    img = cv2.imread('D:/Work/cartificial_intelligence/datasets/VOC/test_licelist.jpg')
    # 调整图片大/
    # img = cv2.resize(img, (620, 480))
    # # # 灰度图
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # # 双边滤波
    # gray = cv2.bilateralFilter(gray, 13, 15, 15)
    # # # 边缘检测
    # edged = cv2.Canny(gray, 30, 200)
    # #
    # # """寻找轮廓（图像矩阵，输出模式，近似方法）"""
    # contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # # 配合上面一句使用：用来兼容cv2和cv3
    # contours = imutils.grab_contours(contours)
    # # # 根据区域大小排序取前十个
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # screenCnt = None
    # # 遍历轮廓，找到车牌轮廓
    # for c in contours:
    #     # 计算轮廓周长（轮廓，是否闭合）
    #     peri = cv2.arcLength(c, True)
    #     # 折线化（轮廓，阈值（越小越接近曲线），是否闭合）返回折线顶点坐标
    #     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    #     # 获取四个顶点（即四边形）
    #     if len(approx) == 4:
    #         screenCnt = approx
    #         break
    # # 如果找到了四边形
    # if screenCnt is not None:
    #     # 根据四个顶点坐标对img画线(图像矩阵，轮廓坐标集，轮廓索引，颜色，线条粗细)
    #     cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
    #
    # print('====> '+ str(screenCnt));


    # return None;
    # return None;
    # """遮罩"""
    # # 创建一个灰度图一样大小的图像矩阵
    # mask = np.zeros(gray.shape, np.uint8)
    # # 将创建的图像矩阵的车牌区域画成白色
    # cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    # # 图像位运算进行遮罩
    # new_image = cv2.bitwise_and(img, img, mask=mask)
    #
    # """图像剪裁"""
    # # 获取车牌区域的所有坐标点
    # (x, y) = np.where(mask == 255)
    # # 获取底部顶点坐标
    # (topx, topy) = (np.min(x), np.min(y))
    # # 获取底部坐标
    # (bottomx, bottomy,) = (np.max(x), np.max(y))
    # # 剪裁
    # Cropped = gray[topx:bottomx, topy:bottomy]
    # cv2.imshow("", img);
    # cv2.waitKey(0);
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 3)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #
    # # 转换为PIL Image对象，并进行OCR识别
    # img_pil = Image.fromarray(gray)
    """OCR识别"""
    text = pytesseract.image_to_string(img, config = '--psm 11');
    # text = pytesseract.image_to_string(img,   lang='chi_sim');#psm 12
    print("车牌结果：",  text);

    # 显示效果
    cv2.imshow('img', img)
    # cv2.imshow('gray', gray)
    # cv2.imshow('edged', edged)
    # cv2.imshow('new_image', Cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()