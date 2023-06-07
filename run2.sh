#!/bin/bash

if [ ! -n "$1" ] ; then
	the_date=`date +%Y%m%d-%H%M%S`
	
	the_dir="save_"${the_date}
else
	the_dir="$1"
fi


echo $the_dir

mkdir -p $the_dir

for num in {1..10}
do
    python main.py
    mv save/ecom-social/predict*.txt $the_dir/
done