#!/bin/sh

python train.py --dataname cora --pseudospx --modelname scn --earlystp --genmasks
python train.py --dataname cora --pseudospx --modelname sccn --earlystp