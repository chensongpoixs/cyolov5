
# lpr3 rest --port 8715 --host 0.0.0.0


# 导入opencv库
import cv2
# 导入依赖包
import hyperlpr3 as lpr3

# 实例化识别对象
catcher = lpr3.LicensePlateCatcher()
# 读取图片
image = cv2.imread("D:/Work/cartificial_intelligence/datasets/VOC/test_licelist.jpg")
# 识别结果
print(catcher(image))
