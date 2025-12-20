@echo off
mkdir attack
mkdir save\models

echo Setting up local libraries...
python setup_libs.py

echo Starting training...
set PYTHONPATH=local_lib;%PYTHONPATH%

:: CAU HINH DATASET
:: Chon dataset ban muon train.

:: Cau hinh cho CIFAR100 (Tu dong download va train)
:: Day la lua chon tot nhat neu ban muon he thong tu tai du lieu ve va chay luon
set DATASET=cifar100
set DATA_ROOT=./data

:: Cau hinh cho TinyImageNet (Can co san du lieu)
:: set DATASET=tinyImageNet
:: set DATA_ROOT=D:\Datasets\tiny-imagenet-200

echo Dang chay training voi dataset: %DATASET% tai %DATA_ROOT%
python PMG_AFT.py --dataset %DATASET% --root %DATA_ROOT% --name demo_run --epochs 10 --batch_size 32
pause
