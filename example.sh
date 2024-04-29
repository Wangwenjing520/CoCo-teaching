#!/usr/bin/env bash

python main_argue.py --noise_type symmetric --noise_rate 0.8 --dataset cifar100 --alpha 6
python main_argue.py --noise_type symmetric --noise_rate 0.5 --dataset cifar100 --alpha 6
python main_argue.py --noise_type symmetric --noise_rate 0.2 --dataset cifar100 --alpha 6