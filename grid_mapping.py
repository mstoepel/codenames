#!/usr/bin/python
# vim: set ts=2 expandtab:
"""
Based on find_subimage code by:
John O'Neil
https://github.com/johnoneil/subimage
"""

import numpy as np
import scipy.ndimage as sp
import scipy.misc
import cv2
import pandas as pd
from PIL import Image, ImageFilter

def find_subimages(primary, subimage, confidence=0.80):
  primary_edges = cv2.Canny(primary, 250, 300, apertureSize=3)  #32, 128
  subimage_edges = cv2.Canny(subimage, 250, 300, apertureSize=3)  #32, 128
  cv2.imwrite('primary_edges.tmp.png',primary_edges)
  cv2.imwrite('subimage_edges.tmp.png',subimage_edges)

  result = cv2.matchTemplate(primary_edges, subimage_edges, cv2.TM_CCOEFF_NORMED) #TM_CCORR_NORMED also works well
  (y, x) = np.unravel_index(result.argmax(),result.shape)

  result[result>=confidence]=1.0
  result[result<confidence]=0.0

  ccs = get_connected_components(result)
  return correct_bounding_boxes(subimage, ccs)  


def cc_shape(component):
  x = component[1].start
  y = component[0].start
  w = component[1].stop-x
  h = component[0].stop-y
  return (x, y, w, h)

def correct_bounding_boxes(subimage, connected_components):
  (image_h, image_w)=subimage.shape[:2]
  corrected = []
  for cc in connected_components:
    (x, y, w, h) = cc_shape(cc)
    presumed_x = x+w/2
    presumed_y = y+h/2
    corrected.append((slice(presumed_y, presumed_y+image_h), slice(presumed_x, presumed_x+image_w)))
  return corrected

def get_connected_components(image):
  s = sp.morphology.generate_binary_structure(2,2)
  labels,n = sp.measurements.label(image)#,structure=s)
  objects = sp.measurements.find_objects(labels)
  return objects

def draw_bounding_boxes(img,connected_components,max_size=0,min_size=0,color=(255,255,255),line_size=2):
  for component in connected_components:
    if min_size > 0 and area_bb(component)**0.5<min_size: continue
    if max_size > 0 and area_bb(component)**0.5>max_size: continue
    (ys,xs)=component[:2]
    cv2.rectangle(img,(xs.start,ys.start),(xs.stop,ys.stop),color,line_size)

def save_output(infile, outfile, connected_components):
  img = cv2.imread(infile)
  draw_bounding_boxes(img, connected_components)
  cv2.imwrite(outfile, img)

def  find_subimages_from_files(primary_image_filename, subimage_filename, confidence):
  '''
  2d cross correlation we'll run requires only 2D images (that is color images
  have an additional dimension holding parallel color channel info). So we 'flatten'
  all images loaded at this time, in effect making them grayscale.
  There is certainly a lot of info that will be lost in this process, so a better approach
  (running separately on each channel and combining the cross correlations?) is probably
  necessary.  
  '''
  primary = cv2.imread(primary_image_filename, cv2.IMREAD_COLOR)
  # primary = cv2.GaussianBlur(primary,(5,5),0)

  img_bw = 255 * (cv2.cvtColor(primary, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')

  se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
  mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

  mask = np.dstack([mask, mask, mask]) / 255
  primary = img * mask

  subimage = cv2.imread(subimage_filename, cv2.IMREAD_COLOR)

  img_bw = 255 * (cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')

  se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
  mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

  mask = np.dstack([mask, mask, mask]) / 255
  img2 = cv2.imread(subimage_filename)
  subimage = img2 * mask

  cv2.imwrite('primary_read.png',primary)
  cv2.imwrite('subimage_read.png',subimage)
  return find_subimages(primary, subimage, confidence)

def main_red():
  primary_image_filename = 'red_only.png'
  subimage_filename = 'red2.png'
  outfile = 'out_red.png'

  image_locations = find_subimages_from_files(primary_image_filename, subimage_filename,confidence=0.1805,)

  save_output(primary_image_filename, outfile, image_locations)
  return image_locations

def main_blue():
  primary_image_filename = 'blue_only.png'
  subimage_filename = 'blue.png'
  outfile = 'out_blue.png'

  image_locations = find_subimages_from_files(primary_image_filename, subimage_filename, confidence=0.22, )

  save_output(primary_image_filename, outfile, image_locations)
  return image_locations

def calc_avg_x(coords):
    avg_x = np.average([coords[0],coords[2]])
    return avg_x

def calc_avg_y(coords):
    avg_y = np.average([coords[1],coords[3]])
    return avg_y

def calc_word_xy(coords):
  "Calculates average x and y for each word"
  coords_list = coords
  x = [calc_avg_x(coords_list[i]) for i in range(len(coords))]
  y = [calc_avg_y(coords_list[i]) for i in range(len(coords))]
  return x, y

def assign_word_num(col, row):
  "Uses rules for col row combinations to determine word number."
  word_num = [int(num_map['WORD_NUM'][num_map['COL'] == col[i]][num_map['ROW'] == row[i]]) for i in
              range(len(col))]
  return word_num

if __name__ == '__main__':

  img = cv2.imread('Images\image2.JPG')
  im_color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  lower = np.array([50, 50, 150]) # 50 50 150
  upper = np.array([255, 255, 255]) #250 250 250

  lower_red = np.array([175, 150, 150])
  upper_red = np.array([255, 255, 255])

  lower_blue = np.array([75, 50, 50])
  upper_blue = np.array([130, 255, 255])

  # mask3 = cv2.inRange(im_color, lower, upper)
  # res3 = cv2.bitwise_and(img, img, mask=mask3)
  # cv2.imwrite('out_test.png', res3)

  mask_red = cv2.inRange(im_color, lower_red, upper_red)
  res_red = cv2.bitwise_and(img, img, mask=mask_red)
  cv2.imwrite('red_only.png', res_red)

  mask_blue = cv2.inRange(im_color, lower_blue, upper_blue)
  res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
  cv2.imwrite('blue_only.png', res_blue)

  image_locations_blue = main_blue()
  image_locations_red = main_red()

  red = [[int(image_locations_red[i][1].start), int(image_locations_red[i][0].start), int(image_locations_red[i][1].stop), int(image_locations_red[i][0].stop)] for i in range(len(image_locations_red))]
  blue = [[int(image_locations_blue[i][1].start), int(image_locations_blue[i][0].start), int(image_locations_blue[i][1].stop), int(image_locations_blue[i][0].stop)] for i in range(len(image_locations_blue))]

  red_x,red_y = calc_word_xy(red)
  blue_x,blue_y = calc_word_xy(blue)

  red_col = list(pd.cut(red_x, 5).codes)  # Uses avg x coords to put each word into one of 5 column categories
  red_row = list(pd.cut(red_y, 5).codes)  # Uses avg y coords to put each word into one of 5 row categories

  blue_col = list(pd.cut(blue_x, 5).codes)  # Uses avg x coords to put each word into one of 5 column categories
  blue_row = list(pd.cut(blue_y, 5).codes)  # Uses avg y coords to put each word into one of 5 row categories

  # PROBLEM - BECAUSE DOING CUTS ONLY ON BLUE, OR RED, IF A COLOR DOESN'T SPAN ALL COLUMNS, THERE WILL BE ISSUES WITH COLUMN ASSIGNMENT.
  # NEED TO SOMEHOW USE FULL COLOR GRID FOR CUTS, OR GIVE RANGE FOR CUT. like max value for either one

  # from scipy.stats import binned_statistic
  # blue_col = list(binned_statistic(blue_x, blue_x, bins=5, range=(min(red_x),max(red_x))))
  # blue_row = list(binned_statistic(blue_y, blue_y, bins=5, range=(min(red_y),max(red_y))))

  num_map = pd.read_csv('col_row_to_word_num.csv')

  red_word_num = set(sorted(assign_word_num(red_col, red_row)))
  blue_word_num = set(sorted(assign_word_num(blue_col, blue_row)))

  ## Coordinates are in (x1,y1,x2,y2) format

  print('Red words are located at: {}'.format(list(red_word_num)))
  print('Blue words are located at: {}'.format(list(blue_word_num)))