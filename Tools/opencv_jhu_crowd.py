



# 使用 OpenCV 可以很方便地在图片上绘制矩形框（bounding box）来标注物体位置。下面是一个简单的示例代码：
#
# python
import cv2
from pathlib import Path
import  os;

from PIL import Image
from tqdm import tqdm

# # 读取图片
# img = cv2.imread('image.jpg')
#
# # 定义矩形框左上角和右下角的坐标
# x1, y1 = 100, 100 # 左上角坐标
# x2, y2 = 200, 200 # 右下角坐标
#
# # 绘制矩形框，参数分别为：图像、左上角坐标、右下角坐标、颜色（BGR格式）、线宽度
# cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), thickness=2)
#
# # 显示结果
# cv2.imshow('result', img)
# cv2.waitKey(0)
# 这段代码将读入一张名为 image.jpg 的图片，并在其中绘制一个左上角坐标为 (100,100)，右下角坐标为 (200,200) 的红色矩形框。




# def visdrone2yolo(dir):

  # print(dir);
  # def convert_box(size, box):
  #     # Convert VisDrone box to YOLO xywh box
  #     dw = 1. / size[0]
  #     dh = 1. / size[1]
  #     # 每张图片对应的txt文件中，数据格式是：cls_id x y w h 其中坐标(x,y)是中心点坐标，并且是相对于图片宽高的比例值 ，并非绝对坐标。
  #     #
  #     # 新版本的yolov5中已经集成了训练visdrone数据集的配置文件，其中附带了数据集的处理方式，主要
  #     #print(str(box) + " --- x = " +  str((box[0] + box[2] / 2)) + ",  y = " +  str(box[1] + box[3]   / 2));
  #     return  box[0]    * dw,  box[1]  * dh, box[2] * dw, box[3] * dh
      # return (( box[0]    + (box[2]  / 2))) * dw, ( box[1] + (box[3]   / 2)) * dh, box[2] * dw, box[3] * dh;

  ##print(dir);
  ##return;

  # (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
  # pbar = tqdm((dir / str('gt')).glob('*.txt'), desc=f'Converting {dir}')
  # for f in pbar:
  #     img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size

      # print(img_size);

      #106 114 24 25 1 0
      # lines = [];
      # path = 'D:/Work/cartificial_intelligence/datasets/JHU-CROWD/jhu_crowd_v2.0/train'
# data_txt =  "D:/Work/cartificial_intelligence/datasets/JHU-CROWD/jhu_crowd_v2.0/train/gt/3628.txt";
# data_img =   "D:/Work/cartificial_intelligence/datasets/JHU-CROWD/jhu_crowd_v2.0/train/images/3628.jpg";
# img = cv2.imread(data_img)
# new file name os.path.splitext(filename)[0]

# with open(data_txt, 'r') as file:  # read annotation.txt
  # print('[warr][file = ' + str(file) + '[f.name = ' + str(dir / 'images' / f.name));
  # 读取图片

  # for row in [x.split(' ') for x in file.read().strip().splitlines()]:


      # 定义矩形框左上角和右下角的坐标
      # x1, y1 = int(int(row[0]) - int(row[2])/2), int(int(row[1]) - int(row[3])/2 ) ; # 左上角坐标
      # x2, y2 = int(int(row[0]) + int(row[2])/2), int(int(row[1]) + int(row[3])/2) ; # 右下角坐标
      # p1, p2 = 1, 1;
      # 绘制点，参数分别为：图像、坐标、半径、颜色（BGR格式）、线宽度
      # cv2.circle(img, (int(row[0]), int(row[1])), radius=3, color=(0, 0, 255), thickness=-1)
      # 绘制矩形框，参数分别为：图像、左上角坐标、右下角坐标、颜色（BGR格式）、线宽度
      # cv2.rectangle(img, (int(row[0]), int(row[1])), (p1, p2), (0, 0, 255), thickness=2)
      # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

      # 显示结果
# cv2.imshow('result', img)
# cv2.waitKey(0)
              #if row[4] == '0':  # VisDrone 'ignored regions' class 0
              #    continue
              # cls = 0;
              # box = convert_box(img_size, tuple(map(int, row[:4])))
              # print(str(row) + " --- " + str(box));
              # lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

              # fcount = 0;
              # for fx in box :
              #     fcount += fx;
              # print(img_size);
              # if fcount > 1.:
              #     print(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n");
              #     print('[warr][file = '+str(file)+ '[f.name = ]'+str(f.name)+ str(img_size)+ '---->fcount = '+ str(fcount) + '] [row = '+str(row)+']');
             #else:
             #    print(f"info-->{cls} {' '.join(f'{x:.6f}' for x in box)}\n");
#

              # with open(str(f).replace(os.sep + 'gt' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
              #     fl.writelines(lines)  # write label.txt


crow_cls = "train";

def visdrone2yolo(dir):
  from PIL import Image
  from tqdm import tqdm

  print(dir);
  # def convert_box(size, box):
  #     # Convert VisDrone box to YOLO xywh box
  #     dw = 1. / size[0]
  #     dh = 1. / size[1]
  #     # 每张图片对应的txt文件中，数据格式是：cls_id x y w h 其中坐标(x,y)是中心点坐标，并且是相对于图片宽高的比例值 ，并非绝对坐标。
  #     #
  #     # 新版本的yolov5中已经集成了训练visdrone数据集的配置文件，其中附带了数据集的处理方式，主要
  #     #print(str(box) + " --- x = " +  str((box[0] + box[2] / 2)) + ",  y = " +  str(box[1] + box[3]   / 2));
  #     return  box[0]    * dw,  box[1]  * dh, box[2] * dw, box[3] * dh
      # return (( box[0]    + (box[2]  / 2))) * dw, ( box[1] + (box[3]   / 2)) * dh, box[2] * dw, box[3] * dh;

  ##print(dir);
  ##return;

  (dir / 'new_images').mkdir(parents=True, exist_ok=True)  # make labels directory
  pbar = tqdm((dir / str('gt')).glob('*.txt'), desc=f'Converting {dir}')
  for f in pbar:
      # img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size

      filename = os.path.basename(f.name);
      new_file = os.path.splitext(filename)[0] + ".jpg";
      # print(img_size);
      img = cv2.imread('D:/Work/cartificial_intelligence/datasets/JHU-CROWD/jhu_crowd_v2.0/'+crow_cls+'/images/'+ new_file);
      #106 114 24 25 1 0
      lines = []
      with open(f, 'r') as file:  # read annotation.txt
          # print('[warr][file = ' + str(file) + '[f.name = ' + str(dir / 'images' / f.name));
          for row in [x.split(' ') for x in file.read().strip().splitlines()]:
              x1, y1 = int(int(row[0]) - int(row[2])/2), int(int(row[1]) - int(row[3])/2 ) ; # 左上角坐标
              x2, y2 = int(int(row[0]) + int(row[2])/2), int(int(row[1]) + int(row[3])/2) ; # 右下角坐标
              p1, p2 = 1, 1;
              # 绘制点，参数分别为：图像、坐标、半径、颜色（BGR格式）、线宽度
              # cv2.circle(img, (int(row[0]), int(row[1])), radius=3, color=(0, 0, 255), thickness=-1)
              # 绘制矩形框，参数分别为：图像、左上角坐标、右下角坐标、颜色（BGR格式）、线宽度
              # cv2.rectangle(img, (int(row[0]), int(row[1])), (p1, p2), (0, 0, 255), thickness=2)
              if int(row[4]) == 1:
                  if int(row[5]) == 0:
                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=1);
                  else:
                      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2);
              elif int(row[4]) == 2:
                  print('-------------------------->')
                  if int(row[5]) == 0:
                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1);
                  else:
                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2);
              elif int(row[4]) == 3:
                  print('------------------------==-->')
                  if int(row[5]) == 0:
                     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1);
                  else:
                     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
              else:
                  cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
      cv2.imwrite( ('D:/Work/cartificial_intelligence/datasets/JHU-CROWD/jhu_crowd_v2.0/'+crow_cls+'/new_images/' + new_file), img);

              #if row[4] == '0':  # VisDrone 'ignored regions' class 0
              #    continue
              # cls = 0;
              # box = convert_box(img_size, tuple(map(int, row[:4])))
              # # print(str(row) + " --- " + str(box));
              # lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
              #
              # fcount = 0;
              # for fx in box :
              #     fcount += fx;
              # print(img_size);
              # if fcount > 1.:
              #     print(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n");
              #     print('[warr][file = '+str(file)+ '[f.name = ]'+str(f.name)+ str(img_size)+ '---->fcount = '+ str(fcount) + '] [row = '+str(row)+']');
             #else:
             #    print(f"info-->{cls} {' '.join(f'{x:.6f}' for x in box)}\n");
#

              # with open(str(f).replace(os.sep + 'gt' + os.sep, os.sep + 'new_image' + os.sep), 'w') as fl:
              #     fl.writelines(lines)  # write label.txt


# Download
dir = Path('D:/Work/cartificial_intelligence/datasets/JHU-CROWD/jhu_crowd_v2.0') ; # dataset root dir
#urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip' ]
#download(urls, dir=dir, curl=True, threads=4)
#dir = Path('./test/jhu_crowd') ; # dataset root dir
print(dir);
#visdrone2yolo(dir / 'train');
#visdrone2yolo(dir  );
# Convert
for d in crow_cls, '---':
   visdrone2yolo(dir / d);  # convert VisDrone annotations to YOLO labels




   # 、JHU_CROWD + +
   # 共计4372张图片，其中训练集2272张，验证集500张，测试集1600张；
   # 图片格式为.jpg，标签格式为.txt；
   # 图片平均分辨率为910 × 1430
   # 910 \times
   # 1430910×1430：
   # 训练集中最小图片尺寸为169 × 117
   # 169 \times
   # 117169×117(2660.j
   # pg)或者222 × 107
   # 222 \times
   # 107222×107(1344.j
   # pg)，最大图片尺寸为8580 × 4089
   # 8580 \times
   # 40898580×4089(1243.j
   # pg)或者7371 × 4914
   # 7371 \times
   # 49147371×4914(1227.j
   # pg)；
   # 训练集中含有11
   # 1111
   # 张灰度图片和59
   # 5959
   # 张竖屏图片；
   # 验证集中最小图片尺寸为300 × 208
   # 300 \times
   # 208300×208(1325.j
   # pg)或者750 × 206
   # 750 \times
   # 206750×206(0179.j
   # pg)，最大图片尺寸为7295 × 1878
   # 7295 \times
   # 18787295×1878(1614.j
   # pg)或者5760 × 3840
   # 5760 \times
   # 38405760×3840(3815.j
   # pg)；
   # 验证集中含有2
   # 22
   # 张灰度图片和19
   # 1919
   # 张竖屏图片；
   # 测试集中最小图片尺寸为232 × 378
   # 232 \times
   # 378232×378(0202.j
   # pg)或者500 × 130
   # 500 \times
   # 130500×130(4271.j
   # pg)，最大图片尺寸为10088 × 3520
   # 10088 \times
   # 352010088×3520(4343.j
   # pg)或者3840 × 5760
   # 3840 \times
   # 57603840×5760(1670.j
   # pg)；
   # 测试集中含有11
   # 1111
   # 张灰度图片和47
   # 4747
   # 张竖屏图片；
   # 训练集、验证集和测试集目录下均包含2
   # 22
   # 个子目录（images，gt），以及一个文件image_labels.txt；
   # images目录下包含图像；
   # gt目录下包含每张图像对应的.txt格式的标签，每个txt文件包含若干行，每一行有6个值x, y, w, h, o, b
   # x, y, w, h, o, bx, y, w, h, o, b，以空格’ '分割：
   # x, y
   # x, yx, y表示头部位置；
   # w, h
   # w, hw, h表示头部的大致宽度和高度;
   # o
   # oo表示遮挡等级，其取值可为1, 2, 3
   # 1, 2, 31, 2, 3，分别表示
   # 可见、部分遮挡、全遮挡；
   # b
   # bb表示模糊登记，其取值可为0, 1
   # 0, 10, 1，分别表示不模糊、模糊；
   # 一个典型示例为
   # 133
   # 229
   # 11
   # 17
   # 2
   # 0；
   # image_labels.txt文件是图像级别的注释，其每一行是对一张图像的注释，具体地，一行包含五个值，以逗号’, '分割：
   # 图片文件名；
   # 图片中总人数；
   # 场景样式，如
   # 会议、街景、火车站、游行等；
   # 天气条件，其取值可为0, 1, 2, 3
   # 0, 1, 2, 30, 1, 2, 3，分别表示
   # 无特殊天气、雾霾、下雨、下雪；
   # 图像是否含有干扰，取值为0
   # 00
   # 表示不含有，取值为1
   # 11
   # 表示含有干扰（图中无人，为负样本；或者图中虽然有人，但背景纹理与人群相似）；
   # 一个典型示例为
   # 00
   # 92, 210, railway
   # station, 0, 0；
   #