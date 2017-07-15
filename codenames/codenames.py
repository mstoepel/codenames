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
            coords.append([line['x1'],line['x2'],line['y1'],line['y2']])

    words_df = pd.DataFrame({'Word':word_list, 'Coords':coords},columns=['Word','Coords','Team','Word_Num'])



    # print output
    print(words_df)
    # print(word_list)
    # print(coords)




