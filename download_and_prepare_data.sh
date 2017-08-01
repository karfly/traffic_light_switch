#!/bin/bash

DATA_URL=http://dl.dropboxusercontent.com/s/ytguyvc7rh8dkjz/traffic_light_switch_data.zip?dl=1
VAL_RATIO=0.2

curl -L -o traffic_light_switch_data.zip $DATA_URL
unzip traffic_light_switch_data.zip
rm traffic_light_switch_data.zip

python3 prepare_data.py --videos-dir=./data/train/videos/ --prepared-data-dir=./data/train/ --val-ratio=$VAL_RATIO
python3 prepare_data.py --videos-dir=./data/public_test/videos/ --prepared-data-dir=./data/public_test/ --val-ratio=0