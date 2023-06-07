#!/bin/bash

if [ ! -n "$1" ] ; then
	the_date=`date +%Y%m%d-%H%M%S`
	
	the_dir="save_"${the_date}
else
	the_dir="$1"
fi


echo $the_dir

mkdir -p $the_dir

if [ ! -f "$the_dir/predict1.txt" ]; then
	python main.py
	cp save/ecom-social/predict.txt $the_dir/predict1.txt
fi

if [ ! -f "$the_dir/predict2.txt" ]; then
	python main.py
	cp save/ecom-social/predict.txt $the_dir/predict2.txt
fi

# if [ ! -f "$the_dir/predict3.txt" ]; then
# 	python main.py
# 	cp save/ecom-social/predict.txt $the_dir/predict3.txt
# fi

python merge_all_predict_files2.py $the_dir

cp $the_dir/predict.txt save/ecom-social/

python merge_all.py
