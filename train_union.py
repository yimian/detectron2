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

import click

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def pdd_v2_map(name):
    name = name.strip()
    if len(name)== 1 and name.isalpha():
        return 'alpha'
    elif len(name) == 1 and name.isdigit():
        return 'num'
    else:
        return name

def pdd_v1_map(name):
    name = name.strip()
    if name == '汉字':
        return 'chn'
    elif name == '数字':
        return 'num'
    else:
        return 'alpha'

def get_cate_info(task_name):
    pass

TASKS = {
    "pdd_v1_type": {
        "cate_list": ['num', 'chn', 'alpha'],
        "cate_map": pdd_v1_map,
    },
    "pdd_v1_direct": {
        "cate_list": ['left','right','up','down','tilt']
    },
    "pdd_v2_type": {
        "cate_list": ['alpha', 'prism', 'cone', 'table', 'cube', 'cylinder', 'opposite', 'ball', 'num'],
        "cate_map": pdd_v2_map,
    },
    "pdd_v2_direct": {
        "cate_list": ["pos", "side"]
    }
}

class AliasedGroup(click.Group):
    """允许使用command缩写"""
    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        matches = [x for x in self.list_commands(ctx)
                   if x.startswith(cmd_name)]
        if not matches:
            return None
        elif len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])
        ctx.fail('Too many matches: %s' % ', '.join(sorted(matches)))


@click.command(cls=AliasedGroup)
@click.pass_context
def cli(ctx):
    ctx.obj = {}


@cli.command()
@click.option('-t', '--task_name', required=True,  help='根据任务名称加载品类列表')
@click.option('-c', '--config_path', required=True,  help='yaml模型配置文件路径')
@click.option('-o', '--output_path', required=True,  help='输出文件路径')
def train(task_name, config_path, output_path):

    if task_name in ['pdd_v1', 'pdd_v2']:
        # 此时为单一模型训练, 同时读取direct和type两个数据集, 合并结果
        main_cate_map = TASKS[f'{task_name}_type'].get('cate_map', lambda x: x)
        main_cate_list = TASKS[f'{task_name}_type']['cate_list']
        aux_cate_map = TASKS[f'{task_name}_direct'].get('cate_map', lambda x: x)
        aux_cate_list = TASKS[f'{task_name}_direct']['cate_list']

        for d in ['train', 'val']:
            s1_args = [f'datasets/{task_name}_type/{d}', main_cate_map, main_cate_list]
            s2_args = [f'datasets/{task_name}_direct/{d}', aux_cate_map, aux_cate_list]
            DatasetCatalog.register(f"{task_name}_{d}", lambda d=d:get_union_pdd_dicts(s1_args, s2_args))
            MetadataCatalog.get(f"{task_name}_{d}" ).set(thing_classes=main_cate_list, aux_thing_classes=aux_cate_list)

    else:
        assert task_name in TASKS, "请先定义任务的品类列表"
        main_cate_map = TASKS[task_name].get('cate_map', lambda x: x)
        main_cate_list = TASKS[task_name]['cate_list']
        aux_cate_list = None
        for d in ['train', 'val']:
            DatasetCatalog.register(f"{task_name}_{d}", lambda d=d: get_pdd_dicts(f'datasets/{task_name}/{d}', main_cate_map, main_cate_list))
            MetadataCatalog.get(f"{task_name}_{d}").set(thing_classes=main_cate_list)


    # 训练配置
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.DATASETS.TRAIN = (f"{task_name}_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(main_cate_list)  # only has one class (ballon)
    if aux_cate_list is not None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES_AUX = len(aux_cate_list)
        cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeadsV2"
    cfg.OUTPUT_DIR = output_path
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # 评估
    evaluator = COCOEvaluator(f"{task_name}_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, f"{task_name}_val")
    r = inference_on_dataset(trainer.model, val_loader, evaluator)
    with open(os.path.join(cfg.OUTPUT_DIR, 'coco_map.json'), 'w') as cc:
        cc.write(json.dumps(r, ensure_ascii=False))
        cc.write('\n')


@cli.command()
@click.option('-t', '--task_name', required=True,  help='根据任务名称加载品类列表')
@click.option('-c', '--config_path', required=True,  help='yaml模型配置文件路径')
@click.option('-m', '--model_path', required=True,  help='模型文件路径')
@click.option('-st', '--score_threshold', default=0.6, help='分类打分阈值')
@click.option('-o', '--output_path', help='结果保存路径')
@click.argument("img_path")
def inference(task_name, config_path, model_path, score_threshold, output_path, img_path):
    # 测试
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = os.path.join(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold  # set the testing threshold for this model
    if task_name in ['pdd_v1', 'pdd_v2']:
        main_cate_list = TASKS[f'{task_name}_type']['cate_list']
        aux_cate_list = TASKS[f'{task_name}_direct']['cate_list']
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(main_cate_list)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES_AUX = len(aux_cate_list)
        cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeadsV2"
    else:
        assert task_name in TASKS
        main_cate_list = TASKS[task_name]['cate_list']
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(main_cate_list)
        aux_cate_list = None

    click.echo('正在加载模型...')
    predictor = DefaultPredictor(cfg)

    rets = []
    if os.path.isdir(img_path):
        for image_name in os.listdir(img_path):
            if image_name.endswith('png') or image_name.endswith('jpeg'):
                im = cv2.imread(os.path.join(img_path, image_name))
                outputs = predictor(im)
                raw_outputs = outputs['instances'].to("cpu")
                rets.append({'filename': image_name, 'result': convert_result(raw_outputs, main_cate_list, aux_cate_list)})

    else:
        im = cv2.imread(img_path)
        outputs = predictor(im)
        raw_outputs = outputs['instances'].to("cpu")
        rets.append({'filename': img_path, 'result': convert_result(raw_outputs, main_cate_list, aux_cate_list)})

    if output_path is not None:
        with open(output_path, 'w') as writer:
            for r in rets:
                writer.write(json.dumps(r, ensure_ascii=False))
                writer.write('\n')


def convert_result(raw_output, cate_list=None, aux_cate_list=None):
    """将模型输出原始结果转换为json"""
    tmp = []
    if aux_cate_list is None:
        for box, score, label in zip(*raw_output._fields.values()):
            box = box.tolist()
            tmp.append({
                'cord': box,
                'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                'class': cate_list[label],
                'score': score.item()
            })
    else:
        for box, score, label, aux_score, aux_label  in zip(*raw_output._fields.values()):
            box = box.tolist()
            tmp.append({
                'cord': box,
                'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                'class': cate_list[label],
                'score': score.item(),
                'class_aux': aux_cate_list[aux_label],
                'score_aux': aux_score.item()
            })
    print('result is', tmp)
    return tmp


def get_pdd_dicts(img_dir, cate_map, cate_list):
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
                obj = {
                    "bbox": list(anno['bndbox'].values()),
                    "bbox_mode": BoxMode.XYXY_ABS,
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
    cli(obj={})

