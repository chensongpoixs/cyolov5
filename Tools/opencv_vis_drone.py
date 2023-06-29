# 使用 OpenCV 可以很方便地在图片上绘制矩形框（bounding box）来标注物体位置。下面是一个简单的示例代码：
#
# python
import cv2
from pathlib import Path
import  os;

from PIL import Image
from tqdm import tqdm




crow_cls = "VisDrone2019-DET-train";



class_name = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'];



# print(class_name[3]);



def visdrone2yolo(dir):
  from PIL import Image
  from tqdm import tqdm

  print(dir);


  (dir / 'new_images').mkdir(parents=True, exist_ok=True)  # make labels directory
  pbar = tqdm((dir / str('annotations')).glob('*.txt'), desc=f'Converting {dir}')
  for f in pbar:
      # img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size

      filename = os.path.basename(f.name);
      new_file = os.path.splitext(filename)[0] + ".jpg";
      # print(img_size);
      img = cv2.imread('D:/Work/cartificial_intelligence/datasets/VisDrone/'+crow_cls+'/images/'+ new_file);
      #106 114 24 25 1 0
      lines = []
      with open(f, 'r') as file:  # read annotation.txt
          # print('[warr][file = ' + str(file) + '[f.name = ' + str(dir / 'images' / f.name));
          for row in [x.split(',') for x in file.read().strip().splitlines()]:
              if row[4] == '0':
                  continue;
              x1, y1 = int(int(row[0]) ), int(int(row[1])  ) ; # 左上角坐标
              x2, y2 = int(int(row[0]) + int(row[2])), int(int(row[1]) + int(row[3])) ; # 右下角坐标
              # 绘制点，参数分别为：图像、坐标、半径、颜色（BGR格式）、线宽度
              # cv2.circle(img, (int(row[0]), int(row[1])), radius=3, color=(0, 0, 255), thickness=-1)
              # 绘制矩形框，参数分别为：图像、左上角坐标、右下角坐标、颜色（BGR格式）、线宽度
              # cv2.rectangle(img, (int(row[0]), int(row[1])), (p1, p2), (0, 0, 255), thickness=2)
             # cv2.putText(img, "Hello ", Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 4, 8);
              cv2.putText(img, class_name[int(row[5]) -1 ], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX   , 0.5, (0, 0, 255), 1, 8);
              cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2);
      cv2.imwrite( ('D:/Work/cartificial_intelligence/datasets/VisDrone/'+crow_cls+'/new_images/' + new_file), img);




# Download
dir = Path('D:/Work/cartificial_intelligence/datasets/VisDrone') ; # dataset root dir
#urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip' ]
#download(urls, dir=dir, curl=True, threads=4)
#dir = Path('./test/jhu_crowd') ; # dataset root dir
print(dir);

for d in crow_cls, '---':
   visdrone2yolo(dir / d);  # convert VisDrone annotations to YOLO labels


