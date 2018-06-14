import os,sys
sys.path.append('.')
import cv2
import numpy as np

from dataset import load_single_image_annotation, load_annotations, draw_bbox

DATA_DIR = 'unit_test/test_data'
OUTPUT_DIR = 'unit_test/test_output'
XML_FNAME = 'unit_test/test_data/2007_000032.xml'
IMG_FNAME = 'unit_test/test_data/2007_000032.jpg'
VOC_FNAME = 'unit_test/test_data/voc.txt'
TMALL_FNAME = 'unit_test/test_data/tmall.txt'

GT_ATTRS = ['aeroplane', 'aeroplane', 'person', 'person']
GT_BBOXES = [[103., 77., 374., 182.],
             [132., 87., 196., 122.],
             [194., 179., 212., 228.],
             [25., 188., 43., 237.]]


def test_load_annotations(filename, zero_based, prefix):
  """test VOC/aergia format annotations"""
  data = load_annotations(filename, zero_based)
  n_data = len(data)
  for i in xrange(n_data):
    output_fname = os.path.join(OUTPUT_DIR, "{}_{}.jpg".format(prefix, i))
    img, bboxes, attrs_list = data[i]
    draw_bbox(img, bboxes, map(lambda x: x[0], attrs_list))
    write_successful = cv2.imwrite(output_fname, img)
    assert write_successful, \
      'fail to write image {} to file'.format(output_fname)


if __name__ == '__main__':
  
  # remove all data in test_output
  cwd_dir = os.getcwd()
  os.chdir(OUTPUT_DIR)
  fnames = os.listdir('.')
  for fname in fnames:
    os.remove(fname)
  os.chdir(cwd_dir)

  # test load_single_image_annotation
  bboxes, attrs_list = load_single_image_annotation(XML_FNAME, zero_based=True)
  print "==============================="
  print "bboxes (xmin, ymin, xmax, ymax)"
  print "gt:"
  for i in xrange(len(GT_BBOXES)):
    print GT_ATTRS[i], tuple(GT_BBOXES[i])
  print "\ntest:"
  for i in xrange(len(bboxes)):
    print ', '.join(attrs_list[i]), tuple(bboxes[i])

  # test draw_bbox
  img = cv2.imread(IMG_FNAME)
  draw_bbox(img, GT_BBOXES, GT_ATTRS)
  output_fname = os.path.join(OUTPUT_DIR, "single.jpg")
  cv2.imwrite(output_fname, img)

  # test voc load_annotations
  test_load_annotations(VOC_FNAME, zero_based=True, prefix='voc')
  test_load_annotations(TMALL_FNAME, zero_based=False, prefix='tmall')
