# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    LOGGER.info('[save_img = '+str(save_img)+']');
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    LOGGER.info('[run] [weights = '+str(weights)+'][ddn =' + str(dnn) + '][data = '+ str(data) +'][fp16 = '+ str(half) +']');

    # Load model
    ## 选择设备，如果device为空，则自动选择设备，如果device不为空，则选择指定的设备
    device = select_device(device)
    #@# 加载模型，DetectMultiBackend()函数用于加载模型，weights为模型路径，device为设备，dnn为是否使用opencv dnn，data为数据集，fp16为是否使用fp16推理
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half);
    # LOGGER.info(model);
    #@# 获取模型的stride，names，pt,model.stride为模型的stride，model.names为模型的类别，model.pt为模型的类型
    stride, names, pt = model.stride, model.names, model.pt
    LOGGER.info('========= [imgsz = '+str(imgsz)+'][stride ' + str(stride) + ']');
    # check image size,验证图像大小是每个维度的stride=32的倍数
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    LOGGER.info('[imgsz = '+str(imgsz)+']');




    # Dataloader  初始化数据集
    bs = 1  # batch_size
    if webcam:
        # 是否显示图片，如果view_img为True，则显示图片
        view_img = check_imshow(warn=True)
        # 创建LoadStreams()对象，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备，vid_stride为视频帧率
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # batch_size为数据集的长度
        bs = len(dataset)
    elif screenshot: #如果source是截图，则创建LoadScreenshots()对象
        # 创建LoadScreenshots()对象，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        # 创建LoadImages()对象，直接加载图片，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备，vid_stride为视频帧率
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # 初始化vid_path和vid_writer，vid_path为视频路径，vid_writer为视频写入对象
    vid_path, vid_writer = [None] * bs, [None] * bs

##########################################################################################################################################
    # Run inference  @@@ 开始推理
    # warmup，预热，用于提前加载模型，加快推理速度，imgsz为图像大小，如果pt为True或者model.triton为True，则bs=1，否则bs为数据集的长度。3为通道数，*imgsz为图像大小，即(1,3,224,224)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # seen为已推理的图片数量，windows为空列表，dt为时间统计对象
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    LOGGER.info('[dataset = '+str(dataset.__dict__)+']');

    # 遍历数据集，path为图片路径，im为图片，im0s为原始图片，vid_cap为视频读取对象，s为视频帧率
    for path, im, im0s, vid_cap, s in dataset:
        # LOGGER.info('[path = '+str(path)+'][im = '+str(im)+'][im0s = '+ str(im0s) +'][vid_cap = '+  str(vid_cap)+'][s = '+str( s)+']');
        # LOGGER.info('[dt = '+str(dt.__dict__)+']');

        # 开始计时，读取图片时间
        with dt[0]:
            # 将图片转换为tensor，并放到模型的设备上，pytorch模型的输入必须是tensor
            im = torch.from_numpy(im).to(model.device);
            # uint8 to fp16/32，如果模型使用fp16推理，则将图片转换为fp16，否则转换为fp32
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 如果图片维度为3，则添加batch维度
            if len(im.shape) == 3:
                #在前面添加batch维度，即将图片的维度从3维转换为4维，即(3,640,640)转换为(1,3,224,224)，pytorch模型的输入必须是4维的
                im = im[None]  # expand for batch dim

        # Inference 推理
        with dt[1]:
            LOGGER.info('[dt[1]]')
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 推理，results为推理结果
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            LOGGER.info('[dt[2]]')
            #@@@@#$ probabilities,对推理结果进行softmax，得到每个类别的概率
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions 处理预测结果
        # 遍历每张图片,enumerate()函数将pred转换为索引和值的形式，i为索引，det为对应的元素，即每个物体的预测框
        for i, det in enumerate(pred):  # per image
            # 检测的图片数量加1
            seen += 1
            # batch_size >= 1，如果是摄像头，则获取视频帧率
            if webcam:  # batch_size >= 1
                # path[i]为路径列表，ims[i].copy()为将输入图像的副本存储在im0变量中，dataset.count为当前输入图像的帧数
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                # 在打印输出中添加当前处理的图像索引号i，方便调试和查看结果。在此处，如果是摄像头模式，i表示当前批次中第i张图像；否则，i始终为0，因为处理的只有一张图像。
                s += f'{i}: '
            else:
                # 如果不是摄像头，frame为0
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # LOGGER.info('[p = '+str(p)+']'); # //  rtsp_//127.0.0.1/live/chensong
            # 将路径转换为Path对象
            p = Path(p)  # to Path
            #保存图片的路径，save_dir为保存图片的文件夹，p.name为图片名称
            save_path = str(save_dir / p.name)  # im.jpg
            # 保存预测框的路径，save_dir为保存图片的文件夹，p.stem为图片名称，dataset.mode为数据集的模式，如果是image，则为图片，否则为视频
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # LOGGER.info('[txt_path = '+txt_path+'][save_path = '+save_path+']'); #// runs\detect\exp38\chensong

            # 打印输出，im.shape[2:]为图片的大小，即(1,3,224,224)中的(224,224)
            s += '%gx%g ' % im.shape[2:]  # print string
           # LOGGER.info('[s ='+s+']'); // jpeg width x height
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # LOGGER.info( '[gn =' +str(gn) + ']'); # video width x height
            imc = im0.copy() if save_crop else im0  # for save_crop
            # LOGGER.info('[imc =' + str(imc) + ']');
            # Annotator()对象，用于在图片上绘制分类结果，im0为原始图片，example为类别名称，pil为是否使用PIL绘制
            LOGGER.info('[line_thickness = '+str(line_thickness)+']');
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            #LOGGER.info('[annotator =' + str(annotator) + ']');
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    LOGGER.info('linux ---> '); # linux上需要特殊设置
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                LOGGER.info('save_img ---->>>');
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # LOGGER.info('======[s = '+s+'][det = '+str(det)+ ']');
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results 打印结果
    # 每张图片的速度
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t) #打印速度
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else '' #如果save_txt为True，则打印保存的标签数量
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")  #打印保存的路径
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
