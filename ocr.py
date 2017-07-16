from subprocess import call
import os

base_path = os.path.join('..', 'ocropy-master')

call("python {} -n Images\image1.* -o OCR_Images".format(os.path.join(base_path, 'ocropus-nlbin')), shell=True)
call("python {} OCR_Images\*.bin.png".format(os.path.join(base_path, 'ocropus-gpageseg')), shell=True)
call("python {} -m ..\ocropy-master\models\uw3unlv-plus-00100000.pyrnn OCR_Images\*\*.bin.png".format(os.path.join(base_path, 'ocropus-rpred')), shell=True)
call("python {} *\*.bin.png -o words.html".format(os.path.join(base_path, 'ocropus-hocr')), shell=True)