#!/bin/bash

the_date=`date +%Y%m%d-%H%M%S`

the_dir="save_"${the_date}

echo $the_dir

mkdir -p $the_dir

python main.py

cp save/ecom-social/predict.txt $the_dir/predict1.txt

python main.py

cp save/ecom-social/predict.txt $the_dir/predict2.txt

python main.py

cp save/ecom-social/predict.txt $the_dir/predict3.txt

python merge_all_predict_files.py $the_dir

cp $the_dir/predict.txt save/ecom-social/
