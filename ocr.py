from subprocess import call

call("python ..\ocropy-master\ocropus-nlbin -n Images\image1.* -o OCR_Images")
call("python ..\ocropy-master\ocropus-gpageseg OCR_Images\*.bin.png")
call("python ..\ocropy-master\ocropus-rpred -m ..\ocropy-master\models\uw3unlv-plus-00100000.pyrnn OCR_Images\*\*.bin.png")
call("python ..\ocropy-master\ocropus-hocr *\*.bin.png -o words.html")