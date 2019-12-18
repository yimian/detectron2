# -*- coding: utf-8 -*-

# @File    : train_union.py
# @Date    : 2019-12-18
# @Author  : skym

# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode

# 定义catelist

def get_aux_cate_list():
    # v2验证码方向
    aux_cate_list= ['pos','side']
    aux_cate_map = lambda x:x
    return aux_cate_list, aux_cate_map

def get_cate_list():
    # v2验证码类型
    alpha = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    alpha1 = [chr(i) for i in range(ord('a'), ord('z')+1)]
    # dict_v0 = {}.fromkeys(alpha, 'alpha0')
    dict_v1 = {}.fromkeys(alpha+alpha1, 'alpha')
    num = [str(i) for i in range(10)]

    dict_v2 = {}.fromkeys(num, 'num')


    tmp = ['opposite', 'table', 'ball','cylinder', 'cube', 'prism', 'cone']
    dict_v3 = {}
    for i in tmp:
        dict_v3[i]=i

    cate_map ={}
    cate_map.update(dict_v1)
    cate_map.update(dict_v2)
    cate_map.update(dict_v3)
    main_cate_list = list(set(cate_map.values()))
    main_cate_map = lambda x: cate_map.get(x,None)
    return main_cate_list, main_cate_map







# write a function that loads the dataset into detectron2's standard format
def get_pdd_dicts(img_dir, cate_map, cate_list):
    print('img_dir ', img_dir)
    idx = 0
    dataset_dicts = []

    for pic_name in os.listdir(img_dir):
        if pic_name.endswith('png') or pic_name.endswith('jpeg'):
            pic_path = os.path.join(img_dir, pic_name)
            label_path = os.path.join(img_dir, 'outputs', pic_name.split('.')[-2] + '.json')
        else:
            continue

        record = {}

        height, width = cv2.imread(pic_path).shape[:2]

        record["file_name"] = pic_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        if not img_dir.endswith('test'):
            with open(label_path) as f:
                imgs_anns = json.load(f)

            for anno in imgs_anns["outputs"]['object']:
                cname = anno['name'].strip()
                #                 print(f'<{cname}>', cate_map(cname))
                obj = {
                    "bbox": list(anno['bndbox'].values()),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    #                 "segmentation": [poly],
                    "category_id": cate_list.index(cate_map(cname)) if cate_map(cname) in cate_list else 0,
                    "category_name": cate_map(cname),
                    "iscrowd": 0
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def get_union_pdd_dicts(main_args, aux_args):
    s1 = get_pdd_dicts(*main_args)
    s2 = get_pdd_dicts(*aux_args)
    union_s = []
    for anno_index, anno in enumerate(zip(s1, s2)):

        anno1, anno2 = anno
        union_anno = anno1

        file_name1 = anno1['file_name'].split('/')[-1]
        file_name2 = anno2['file_name'].split('/')[-1]
        assert file_name1 == file_name2, '%s, %s'%(file_name1, file_name2)

        tmp_boxes = {}
        for box in anno2['annotations']:
            cord = ((box['bbox'][0]+box['bbox'][2])/2, (box['bbox'][1]+box['bbox'][3])/2)
            tmp_boxes[cord] = box

        for index, box in enumerate(anno1['annotations']):
            cord = ((box['bbox'][0] + box['bbox'][2])/2, (box['bbox'][1]+box['bbox'][3])/2)
            tmp_box = tmp_boxes.get(cord, None)
            if tmp_box is None:
                keys = list(tmp_boxes.keys())

                distances = [(cord[0]-i)*(cord[0]-i)+(cord[1]-j)*(cord[1]-j) for i,j in keys]
                tmp_box = tmp_boxes[keys[distances.index(min(distances))]]

                assert min(distances)<20, 'err, %s, %s, %s'%(file_name1, box['bbox'],tmp_box['bbox'] )
            union_anno['annotations'][index]['aux_category_id'] = tmp_box['category_id']
            union_anno['annotations'][index]['aux_category_name'] = tmp_box['category_name']
        union_s.append(union_anno)
    return union_s


if __name__ == '__main__':
    # 注册训练任务
    task_name = 'pdd_union'
    main_cate_list, main_cate_map = get_cate_list()
    aux_cate_list, aux_cate_map = get_aux_cate_list()
    for d in ["train", "val"]:
        s1_args = [f'datasets/pdd_v2_type/{d}', main_cate_map, main_cate_list]
        s2_args = [f'datasets/pdd_v2_direct/{d}', aux_cate_map, aux_cate_list]
        DatasetCatalog.register(f"{task_name}_" + d, lambda d=d: get_union_pdd_dicts(s1_args, s2_args))
        MetadataCatalog.get(f"{task_name}_" + d).set(thing_classes=main_cate_list, aux_thing_classes=aux_cate_list)
    pdd_metadata = MetadataCatalog.get(f"{task_name}_train")

    # 训练配置
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TRAIN = (f"{task_name}_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.STEPS = (400, 900)
    cfg.SOLVER.MAX_ITER = 1500  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(main_cate_list)  # only has one class (ballon)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES_AUX = len(aux_cate_list)
    cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeadsV2'
    cfg.OUTPUT_DIR = 'pdd_union_out/'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    # trainer.train()

    # 测试
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set the testing threshold for this model
    cfg.DATASETS.TEST = (f"{task_name}_val",)
    predictor = DefaultPredictor(cfg)


    s1_args = ['datasets/pdd_v2_type/val', main_cate_map, main_cate_list]
    s2_args = ['datasets/pdd_v2_direct/val', aux_cate_map, aux_cate_list]
    dataset_dicts = get_union_pdd_dicts(s1_args, s2_args)
    for d in random.sample(dataset_dicts, 1):
        # for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        print(outputs)
    #     v = Visualizer(im[:, :, ::-1],
    #                    metadata=pdd_metadata,
    #                    scale=1,
    #     )
    #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     plt.imshow(v.get_image()[:, :, ::-1])
    #     plt.show()
    #
    #     output_name = d["file_name"].rsplit('/', 1)[1]
    #     output_path = os.path.join(cfg.OUTPUT_DIR, 'type_val', output_name)
    # #     cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
    #
