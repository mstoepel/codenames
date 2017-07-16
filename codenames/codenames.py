"""Extract text from ocropus output.

There should be 25 words in the output. Can use their coordinates to figure out which
word belongs to each team and which word is the auto loss word.


"""

import re
import sys
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np


def get_closest_word(word):
    "This function will get string similarities for each ocr'd word against all possible Codenames words and convert to the best match word if similarity != 1"
    pass

def calc_avg_x(coords):
    avg_x = np.average([coords[0],coords[2]])
    return avg_x

def calc_avg_y(coords):
    avg_y = np.average([coords[1],coords[3]])
    return avg_y

def calc_word_xy(coords):
    "Calculates average x and y for each word"
    coords_list = coords
    x = [calc_avg_x(coords_list[i]) for i in range(len(word_list))]
    y = [calc_avg_y(coords_list[i]) for i in range(len(word_list))]
    return x, y

def assign_word_num(col,row):
    "Uses rules for col row combinations to determine word number."
    word_num = [int(num_map['WORD_NUM'][num_map['COL'] == col[i]][num_map['ROW'] == row[i]]) for i in range(len(word_list))]
    return word_num

def hocr_to_lines(hocr_path):
    lines = []
    soup = BeautifulSoup(hocr_path)
    for tag in soup.select('.ocr_line'):
        m = re.match(r'bbox (-?\d+) (-?\d+) (-?\d+) (-?\d+)',tag.get('title'))
        assert m
        x0, y0, x1, y1 = (int(v) for v in m.groups())
        lines.append({
            'text': tag.text,
            'x1': x0,
            'x2': x1,
            'y1': y0,
            'y2': y1
        })
    return lines

if __name__ == '__main__':
    num_map = pd.read_csv('..\col_row_to_word_num.csv')
    hocr_path = '..\words.html'
    lines = hocr_to_lines(open(hocr_path).read())

    output = ''
    word_list = []
    coords = []
    for idx, line in enumerate(lines):
        if idx > 0:
            output += '\n'
        if 'text' in line:
            line['text'] = str(line['text']).upper()
            output += line['text']
            word_list.append(line['text'])
            coords.append([line['x1'],line['y1'],line['x2'],line['y2']])

    x, y = calc_word_xy(coords)
    col = list(pd.cut(x,5).codes)       #Uses avg x coords to put each word into one of 5 column categories
    row = list(pd.cut(y,5).codes)       #Uses avg y coords to put each word into one of 5 row categories
    word_num = assign_word_num(col,row)
    words_df = pd.DataFrame({'Word':word_list, 'Coords':coords, 'X':x, 'Y':y, 'Column':col, 'Row':row, 'Word_Num':word_num},columns=['Word','Coords','X','Y','Column','Row','Team','Word_Num'])



    # print output
    print(words_df)
    # print(word_list)
    # print(coords)




