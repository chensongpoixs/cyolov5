#import download, os, Path, utils;
# created: 		2023-05-11
#
# author:			chensong
#
# purpose:		_C_DTLS_ _H_
# 输赢不重要，答案对你们有什么意义才重要。
#
# 光阴者，百代之过客也，唯有奋力奔跑，方能生风起时，是时代造英雄，英雄存在于时代。或许世人道你轻狂，可你本就年少啊。 看护好，自己的理想和激情。
#
#
# 我可能会遇到很多的人，听他们讲好2多的故事，我来写成故事或编成歌，用我学来的各种乐器演奏它。
# 然后还可能在一个国家遇到一个心仪我的姑娘，她可能会被我帅气的外表捕获，又会被我深邃的内涵吸引，在某个下雨的夜晚，她会全身淋透然后要在我狭小的住处换身上的湿衣服。
# 3小时候后她告诉我她其实是这个国家的公主，她愿意向父皇求婚。我不得已告诉她我是穿越而来的男主角，我始终要回到自己的世界。
# 然后我的身影慢慢消失，我看到她眼里的泪水，心里却没有任何痛苦，我才知道，原来我的心被丢掉了，我游历全世界的原因，就是要找回自己的本心。
# 于是我开始有意寻找各种各样失去心的人，我变成一块砖头，一颗树，一滴水，一朵白云，去听大家为什么会失去自己的本心。
# 我发现，刚出生的宝宝，本心还在，慢慢的，他们的本心就会消失，收到了各种黑暗之光的侵蚀。
# 从一次争论，到嫉妒和悲愤，还有委屈和痛苦，我看到一只只无形的手，把他们的本心扯碎，蒙蔽，偷走，再也回不到主人都身边。
# 我叫他本心猎手。他可能是和宇宙同在的级别 但是我并不害怕，我仔细回忆自己平淡的一生 寻找本心猎手的痕迹。
# 沿着自己的回忆，一个个的场景忽闪而过，最后发现，我的本心，在我写代码的时候，会回来。
# 安静，淡然，代码就是我的一切，写代码就是我本心回归的最好方式，我还没找到本心猎手，但我相信，顺着这个线索，我一定能顺藤摸瓜，把他揪出来。
#from utils.general
#import download, os, Path

from pathlib import Path
import  os;
#import  Path;
pic_set = set([ ]);


pic_size = [0, 1];

def visdrone2yolo(dir):
  from PIL import Image
  from tqdm import tqdm

  print(dir);
  def convert_box(size, box):
      # Convert VisDrone box to YOLO xywh box
      dw = 1. / size[0]
      dh = 1. / size[1]
      # 每张图片对应的txt文件中，数据格式是：cls_id x y w h 其中坐标(x,y)是中心点坐标，并且是相对于图片宽高的比例值 ，并非绝对坐标。
      #
      # 新版本的yolov5中已经集成了训练visdrone数据集的配置文件，其中附带了数据集的处理方式，主要
      #print(str(box) + " --- x = " +  str((box[0] + box[2] / 2)) + ",  y = " +  str(box[1] + box[3]   / 2));
      return  box[0]    * dw,  box[1]  * dh, box[2] * dw, box[3] * dh
      # return (( box[0]    + (box[2]  / 2))) * dw, ( box[1] + (box[3]   / 2)) * dh, box[2] * dw, box[3] * dh;

  ##print(dir);
  ##return;

  (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
  pbar = tqdm((dir / str('gt')).glob('*.txt'), desc=f'Converting {dir}');




  for f in pbar:
      img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size;
      # if img_size[0] > 1920:
      pic_set.update(set([ img_size]));
      # temp_size = pic_size;
      if img_size[0] > pic_size[0]:
            pic_size[0] = img_size[0];
      #break;
      # print(pic_set);

      #106 114 24 25 1 0
#       lines = []
#       with open(f, 'r') as file:  # read annotation.txt
#           # print('[warr][file = ' + str(file) + '[f.name = ' + str(dir / 'images' / f.name));
#           for row in [x.split(' ') for x in file.read().strip().splitlines()]:
#               #if row[4] == '0':  # VisDrone 'ignored regions' class 0
#               #    continue
#               cls = 0;
#               box = convert_box(img_size, tuple(map(int, row[:4])))
#               # print(str(row) + " --- " + str(box));
#               lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
#
# #               fcount = 0;
# #               for fx in box :
# #                   fcount += fx;
# #               print(img_size);
# #               if fcount > 1.:
# #                   print(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n");
# #                   print('[warr][file = '+str(file)+ '[f.name = ]'+str(f.name)+ str(img_size)+ '---->fcount = '+ str(fcount) + '] [row = '+str(row)+']');
# #              #else:
# #              #    print(f"info-->{cls} {' '.join(f'{x:.6f}' for x in box)}\n");
# # #
#
#               with open(str(f).replace(os.sep + 'gt' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
#                   fl.writelines(lines)  # write label.txt


# Download
dir = Path('D:/Work/cartificial_intelligence/datasets/JHU-CROWD/jhu_crowd_v2.0') ; # dataset root dir
#urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip' ]
#download(urls, dir=dir, curl=True, threads=4)
#dir = Path('./test/jhu_crowd') ; # dataset root dir
print(dir);
#visdrone2yolo(dir / 'train');
#visdrone2yolo(dir  );
# Convert
for d in 'val', 'train', 'test':
   visdrone2yolo(dir / d);  # convert VisDrone annotations to YOLO labels



print(pic_set);

print('size > 2000 ===> ', pic_size);
print(len(pic_set));