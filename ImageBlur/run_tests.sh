#!/bin/bash

make

./ImageBlur -i DatasetW/0/input.ppm -o  Dataset/0/output.ppm 
./ImageBlur -i DatasetW/1/input.ppm -o  Dataset/1/output.ppm 
./ImageBlur -i Dataset/0/input.ppm -o  Dataset/0/output.ppm 
./ImageBlur -i Dataset/1/input.ppm -o  Dataset/1/output.ppm
#./ImageBlur -i Dataset/2/input.ppm -o  Dataset/2/output.ppm has problems reading file
./ImageBlur -i Dataset/3/input.ppm -o  Dataset/3/output.ppm
./ImageBlur -i Dataset/4/input.ppm -o  Dataset/4/output.ppm
./ImageBlur -i Dataset/5/input.ppm -o  Dataset/5/output.ppm
./ImageBlur -i Dataset/6/input.ppm -o  Dataset/6/output.ppm
./ImageBlur -i Dataset/7/input.ppm -o  Dataset/7/output.ppm
./ImageBlur -i Dataset/8/input.ppm -o  Dataset/8/output.ppm
./ImageBlur -i Dataset/9/input.ppm -o  Dataset/9/output.ppm
