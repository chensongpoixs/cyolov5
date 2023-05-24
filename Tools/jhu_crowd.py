#import download, os, Path, utils;

#from utils.general
#import download, os, Path

from pathlib import Path
import  os;
#import  Path;

def visdrone2yolo(dir):
  from PIL import Image
  from tqdm import tqdm

  def convert_box(size, box):
      # Convert VisDrone box to YOLO xywh box
      dw = 1. / size[0]
      dh = 1. / size[1]
      return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh;

  ##print(dir);
  ##return;

  (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
  pbar = tqdm((dir / str('gt')).glob('*.txt'), desc=f'Converting {dir}')
  for f in pbar:
      img_size = Image.open((dir / 'train' / f.name).with_suffix('.jpg')).size
      print(img_size);
      #106 114 24 25 1 0
      lines = []
      with open(f, 'r') as file:  # read annotation.txt
          for row in [x.split(' ') for x in file.read().strip().splitlines()]:
              #if row[4] == '0':  # VisDrone 'ignored regions' class 0
              #    continue
              cls = 0;
              box = convert_box(img_size, tuple(map(int, row[:4])))
              lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
              with open(str(f).replace(os.sep + 'gt' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
                  fl.writelines(lines)  # write label.txt


# Download
dir = Path('./images') ; # dataset root dir
#urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip' ]
#download(urls, dir=dir, curl=True, threads=4)

print(dir);
visdrone2yolo(dir  );
# Convert
#for d in 'images':
 # visdrone2yolo(dir / d);  # convert VisDrone annotations to YOLO labels