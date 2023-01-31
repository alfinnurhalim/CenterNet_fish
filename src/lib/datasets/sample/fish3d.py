from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import cv2
import os
import math

from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import pycocotools.coco as coco

import imgaug.augmenters as iaa

class Fish3dDataset(data.Dataset):

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    img = cv2.imread(img_path)

    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    img_annos = self.coco.loadAnns(ids=ann_ids)
    annos = [self._coco_box_to_bbox(ann['bbox']) for ann in img_annos]

    num_objs = min(len(annos), self.max_objs)
    meta = dict()

    #  IMG AUG
    aug = iaa.Sequential([
              iaa.AddToHueAndSaturation((-40, 30), per_channel=True),
              iaa.AverageBlur(k=(1, 8)),
          ])

    # no aug
    # img = aug(image=img)

    inp = (img.astype(np.float32) / 255.)

    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    # ===========shape============
    hm = np.zeros((self.opt.num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)

    dep = np.zeros((self.max_objs, 1), dtype=np.float32)
    dim = np.zeros((self.max_objs, 3), dtype=np.float32)
    rot = np.zeros((self.max_objs, 4), dtype=np.float32)

    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    # ===========shape============

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    
    for k in range(num_objs):
      ann = img_annos[k]
      bbox = np.array(annos[k])
      bbox = bbox / self.opt.down_ratio

      cls_id = int(self.cat_ids[1])

      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

      # only taking hm where h and w >0
      if h > 0 and w > 0:

        # HEATMAP AND CENTER
        cx_3d = ann['cx']
        cy_3d = ann['cy']
        ct = np.array(
          [cx_3d,cy_3d], dtype=np.float32)/self.opt.down_ratio
        ct_int = ct.astype(np.int32)

        meta['center'] = ct_int
        radius = gaussian_radius((h, w))
        radius = max(0, int(radius))
        
        draw_gaussian(hm[cls_id], ct, radius)

        # 3d prop
        alphaX = ann['alphax'] % (2*np.pi)
        alphaY = ann['alphay'] % (2*np.pi)

        dep[k] = ann['depth']
        dim[k] = ann['dim']
        rot[k] = [np.sin(alphaX),np.cos(alphaX),
                  np.sin(alphaY),np.cos(alphaY)]

        ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
        reg[k] = ct - ct_int

        reg_mask[k] = 1

    ret = {'input': inp, 
            'hm': hm,
            'ind': ind,
            'dep': dep, 
            'dim': dim, 
            'rot':rot,
            'reg_mask': reg_mask,
            'reg' : reg,
            'meta': meta}

    return ret