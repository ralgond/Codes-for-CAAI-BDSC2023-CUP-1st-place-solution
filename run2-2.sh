#!/bin/bash

the_dir=`ls -l . |grep save_ |tail -1 | awk '{print $NF}'`
echo "choose the dir: $the_dir"

args=`python sort_file_by_mr.py $the_dir`

echo $args

python merge_all_predict_files2.py $the_dir $args

python merge_all2.py $the_dir/predict.txt
