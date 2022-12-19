from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math
import random
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import pycocotools.coco as coco

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap

class DddDataset(data.Dataset):

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _bbs_box_to_bbox(self, box):
    bbox = np.array([box.x1, box.y1, box.x2, box.y2],
                    dtype=np.float32)
    return bbox

  def _convert_alpha(self, alpha):
    return math.radians(alpha) if self.alpha_in_degree else alpha

  def _aug_mosaic(self,output_size,scale_range,index):
    random.seed(index)
    # get random img index
    prob = random.uniform(0.0, 1.0)
    idxs = random.sample(range(len(self.images)), 4)
    # rand = random.randint(0,len(self.images))
    # idxs = [rand,rand,rand,rand]
    new_annos = []
    
    # img output placeholder
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    
    # get scale
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0]) if prob > 0.5 else 1
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0]) if prob > 0.5 else 1

    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    # random resizing
    divid_point_x,divid_point_y,scale_x,scale_y
    
    for i, idx in enumerate(idxs):
      img_id = self.images[idx]
      img_info = self.coco.loadImgs(ids=[img_id])[0]
      img_path = os.path.join(self.img_dir, img_info['file_name'])
      img = cv2.imread(img_path)

      ann_ids = self.coco.getAnnIds(imgIds=[img_id])
      img_annos = self.coco.loadAnns(ids=ann_ids)

      if i == 0:  # top-left
        img = cv2.resize(img, (divid_point_x, divid_point_y))
        output_img[:divid_point_y, :divid_point_x, :] = img
        for ann in img_annos:
          bbox = self._coco_box_to_bbox(ann['bbox'])
          xmin = bbox[0] * scale_x
          ymin = bbox[1] * scale_y
          xmax = bbox[2] * scale_x
          ymax = bbox[3] * scale_y
          new_annos.append([xmin, ymin, xmax, ymax])

      elif (i == 1 and prob > 0.5):  # top-right
        img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))
        output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
        for ann in img_annos:
          bbox = self._coco_box_to_bbox(ann['bbox'])
          xmin = divid_point_x + bbox[0] * (1 - scale_x)
          ymin = bbox[1] * scale_y
          xmax = divid_point_x + bbox[2] * (1 - scale_x)
          ymax = bbox[3] * scale_y
          new_annos.append([xmin, ymin, xmax, ymax])

      elif (i == 2 and prob > 0.5):  # bottom-left
        img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
        output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
        for ann in img_annos:
          bbox = self._coco_box_to_bbox(ann['bbox'])
          xmin = bbox[0] * scale_x
          ymin = divid_point_y + bbox[1] * (1 - scale_y)
          xmax = bbox[2] * scale_x
          ymax = divid_point_y + bbox[3] * (1 - scale_y)
          new_annos.append([xmin, ymin, xmax, ymax])

      elif (i == 3 and prob > 0.5):  # bottom-right
        img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
        output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
        for ann in img_annos:
          bbox = self._coco_box_to_bbox(ann['bbox'])
          xmin = divid_point_x + bbox[0] * (1 - scale_x)
          ymin = divid_point_y + bbox[1] * (1 - scale_y)
          xmax = divid_point_x + bbox[2] * (1 - scale_x)
          ymax = divid_point_y + bbox[3] * (1 - scale_y)
          new_annos.append([xmin, ymin, xmax, ymax])
              
    return output_img,new_annos

  def __getitem__(self, index):
    scale_range = (0.4, 0.6)
    output_size = (512,512)

    # only for 2d
    
    # aug_prob = 1
    # if aug_prob > 0.5:
    img,annos = self._aug_mosaic(output_size,scale_range,index)
    # else:
    #   img_id = self.images[index]
    #   img_info = self.coco.loadImgs(ids=[img_id])[0]
    #   img_path = os.path.join(self.img_dir, img_info['file_name'])
    #   img = cv2.imread(img_path)

    #   ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    #   img_annos = self.coco.loadAnns(ids=ann_ids)
    #   annos = [self._coco_box_to_bbox(ann['bbox']) for ann in img_annos]

    num_objs = min(len(annos), self.max_objs)

    # lower_bound = 1
    # upper_bound = 5
    # ratio = random.uniform(lower_bound, upper_bound)

    
    # im_h,im_w,_ = img.shape
    # new_h = int(im_h/ratio)
    # new_w = int(im_w/ratio)

    # aug = iaa.Sequential([
    #         iaa.Multiply((0.5, 1.1)),
    #         iaa.AverageBlur(k=(1, 7)),
    #         iaa.Resize({"height": new_h, "width": new_w}),
    #         iaa.Resize({"height": im_h, "width": im_w}),
    #     ] )

    aug = iaa.Sequential([
              iaa.AddToHueAndSaturation((-40, 30), per_channel=True),
              # iaa.Rot90((1, 3)),
              # iaa.Fliplr(0.5),
              # iaa.Flipud(0.5),
              # iaa.AverageBlur(k=(1, 20)),
              # iaa.Resize({"height": new_h, "width": new_w}),
              # iaa.Resize({"height": im_h, "width": im_w}),
          ])

    # bbox_list = list()

    # for bbox in annos:
    #   x1 = bbox[0]
    #   y1 = bbox[1]
    #   x2 = bbox[2]
    #   y2 = bbox[3]

    #   bbox = BoundingBox(x1=x1,y1=y1,x2=x2,y2=y2)
    #   bbox_list.append(bbox)

    # bbs = BoundingBoxesOnImage(bbox_list,shape=img.shape)
    # img, bbs_aug = aug(image=img, bounding_boxes=bbs)

    # need to uncomment this for 3d
    # if 'calib' in img_info:
    #   calib = np.array(img_info['calib'], dtype=np.float32)
    # else:
    #   calib = self.calib

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
    if self.opt.keep_res:
      s = np.array([self.opt.input_w, self.opt.input_h], dtype=np.int32)
    else:
      s = np.array([width, height], dtype=np.int32)
    
    aug = False
    if self.split == 'train' and np.random.random() < self.opt.aug_ddd:
      aug = True
      sf = self.opt.scale
      cf = self.opt.shift
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      c[0] += img.shape[1] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      c[1] += img.shape[0] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)

    trans_input = get_affine_transform(
      c, s, 0, [self.opt.input_w, self.opt.input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    # if self.split == 'train' and not self.opt.no_color_aug:
    #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    num_classes = self.opt.num_classes
    trans_output = get_affine_transform(
      c, s, 0, [self.opt.output_w, self.opt.output_h])

    hm = np.zeros(
      (num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    dep = np.zeros((self.max_objs, 1), dtype=np.float32)
    rotbin = np.zeros((self.max_objs, 2), dtype=np.int64)
    rotres = np.zeros((self.max_objs, 2), dtype=np.float32)
    dim = np.zeros((self.max_objs, 3), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    rot_mask = np.zeros((self.max_objs), dtype=np.uint8)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    
    gt_det = []

    for k in range(num_objs):
      # ann = anns[k]
      # bbox = self._coco_box_to_bbox(ann['bbox'])
      # bbox_converted = self._bbs_box_to_bbox(bbs_aug[k])
      bbox = np.array(annos[k])

      # cls_id = int(self.cat_ids[ann['category_id']])
      cls_id = int(self.cat_ids[1])
      if cls_id <= -99:
        continue
      # if flipped:
      #   bbox[[0, 2]] = width - bbox[[2, 0]] - 1

      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)

      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((h, w))
        radius = max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        if cls_id < 0:
          ignore_id = [_ for _ in range(num_classes)] \
                      if cls_id == - 1 else  [- cls_id - 2]
          if self.opt.rect_mask:
            hm[ignore_id, int(bbox[1]): int(bbox[3]) + 1, 
              int(bbox[0]): int(bbox[2]) + 1] = 0.9999
          else:
            for cc in ignore_id:
              draw_gaussian(hm[cc], ct, radius)
            hm[ignore_id, ct_int[1], ct_int[0]] = 0.9999
          continue
        draw_gaussian(hm[cls_id], ct, radius)

        wh[k] = 1. * w, 1. * h
        # gt_det.append([k, ct[0], ct[1], bbox, bbox_converted])
        # gt_det.append([ct[0], ct[1], 1] + \
        #               self._alpha_to_8(self._convert_alpha(ann['alphax'])) + \
        #               [ann['depth']] + (np.array(ann['dim']) / 1).tolist() + [cls_id])
        gt_det.append([ct_int[1], ct_int[0]])
        # if self.opt.reg_bbox:
        #   gt_det[-1] = gt_det[-1][:-1] + [w, h] + [gt_det[-1][-1]]
        # if (not self.opt.car_only) or cls_id == 1: # Only estimate ADD for cars !!!

        # alphaX = ann['alphax'] % (2*np.pi)
        # alphaY = ann['alphay'] % (2*np.pi)

        # # BINNING
        # bot_thr = np.radians(30)
        # up_thr = np.radians(150)

        # if alphaX < bot_thr or alphaX > up_thr:
        #     bin_class = 0
        #     rotbin[k, bin_class] = 1
        #     rotres[k, bin_class] = alphaX 

        # if alphaX < -bot_thr or alphaX > -up_thr:
        #     bin_class = 1
        #     rotbin[k, bin_class] = 1
        #     rotres[k, bin_class] = alphaX 

        # dep[k] = ann['depth']
        # dim[k] = ann['dim']

        dep[k] = 0
        dim[k] = [0,0,0]

        # print('        cat dim', cls_id, dim[k])
        ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1 if not aug else 0
        rot_mask[k] = 1
    # print('gt_det', gt_det)
    # print('')
    ret = {'input': inp, 'hm': hm, 'dep': dep, 'dim': dim, 'ind': ind, 
           'rotbin': rotbin, 'rotres': rotres, 'reg_mask': reg_mask,
           'rot_mask': rot_mask}
    # ret = {'input': inp, 'hm': hm, 'dep': 0, 'dim': np.array([0,0,0]), 'ind': ind, 
    #        'rotbin': rotbin, 'rotres': rotres, 'reg_mask': reg_mask,
    #        'rot_mask': rot_mask}
    if self.opt.reg_bbox:
      ret.update({'wh': wh})
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not ('train' in self.split):
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 18), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'calib': 0,
              'image_path': 0, 'img_id': 0,'annos':annos}
      ret['meta'] = meta
    
    return ret

  def _alpha_to_8(self, alpha):
    # return [alpha, 0, 0, 0, 0, 0, 0, 0]
    ret = [0, 0, 0, 1, 0, 0, 0, 1]
    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
      r = alpha - (-0.5 * np.pi)
      ret[1] = 1
      ret[2], ret[3] = np.sin(r), np.cos(r)
    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
      r = alpha - (0.5 * np.pi)
      ret[5] = 1
      ret[6], ret[7] = np.sin(r), np.cos(r)
    return ret
