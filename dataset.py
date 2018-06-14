import cv2
import os 
import numpy as np
import xml.etree.ElementTree as ET


def load_annotations(filename, zero_based):
  """load annotations from dataset"""
  data = []
  with open(filename) as f:
    lines = f.readlines()
    for line in lines:
      img_fname, xml_fname = line.strip().split(' ')
      img = cv2.imread(img_fname)
      assert img is not None, \
        'fail to read image {}'.format(img_fname)
      bboxes, attrs_list = \
          load_single_image_annotation(xml_fname, zero_based)
      data.append([img, bboxes, attrs_list])
  return data


def load_single_image_annotation(filename, zero_based):
  """load VOC/aergia-format detection annotation for one image"""
  tree = ET.parse(filename)
  objs = tree.findall('object')
  attrs_list = []
  bboxes = []
  for obj in objs:
    # the unique bbox
    bbox = obj.find('bndbox')

    if bbox is None:
      continue
    
    xmin = float(bbox.find('xmin').text)
    xmax = float(bbox.find('xmax').text)
    ymin = float(bbox.find('ymin').text)
    ymax = float(bbox.find('ymax').text)

    # Make pixel indexes 0-based
    if zero_based:
      xmin -= 1
      xmax -= 1
      ymin -= 1
      ymax -= 1

    assert xmin >= 0.0 and xmin <= xmax, \
      'Invalid bounding box x-coord xmin {} or xmax {} at {}' \
        .format(xmin, xmax, filename)
    assert ymin >= 0.0 and ymin <= ymax, \
      'Invalid bounding box y-coord ymin {} or ymax {} at {}' \
        .format(ymin, ymax, filename)

    bboxes.append([xmin, ymin, xmax, ymax])

    attrs = \
      [attr.text.lower().strip() for attr in obj.findall('name')]
    attrs_list.append(attrs)
  
  bboxes = np.array(bboxes)

  return bboxes, attrs_list


def draw_bbox(im, bboxes, attrs, color=(0,255,0), cdict=None):
  """draw bbox for image"""
  for bbox, attr in zip(bboxes, attrs):
    xmin, ymin, xmax, ymax = [int(b) for b in bbox]

    # TODO: enable cdict
    c = color

    # draw box, label
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, attr, (xmin, ymax), font, 0.3, c, 1)