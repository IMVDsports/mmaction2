#!/usr/bin/env bash
python fusion.py 2 ./work_dirs/two_stream/ymja_flow_finetuned.json ./work_dirs/two_stream/ymja_rgb_finetuned.json ./confusion_matrices/ymja_fusion_finetuned.png > ./work_dirs/two_stream/ymja_finetuned.txt
python fusion.py 2 ./work_dirs/two_stream/ymja_flow_finetuned.json ./work_dirs/two_stream/ymja_rgb.json ./confusion_matrices/ymja_fusion_rsff.png > ./work_dirs/two_stream/ymja_rsff.txt
python fusion.py 2 ./work_dirs/two_stream/ymja_flow.json ./work_dirs/two_stream/ymja_rgb.json ./confusion_matrices/ymja_fusion.png > ./work_dirs/two_stream/ymja_scratch.txt
python fusion.py 1 ./work_dirs/two_stream/ymja_rgb.json ./confusion_matrices/ymja_rgb.png > ./work_dirs/two_stream/ymja_rgb.txt
python fusion.py 1 ./work_dirs/two_stream/ymja_flow.json ./confusion_matrices/ymja_flow.png > ./work_dirs/two_stream/ymja_flow.txt
python fusion.py 1 ./work_dirs/two_stream/ymja_flow_finetuned.json ./confusion_matrices/ymja_flow_finetuned.png > ./work_dirs/two_stream/ymja_flow_fine.txt
python fusion.py 1 ./work_dirs/two_stream/ymja_rgb_finetuned.json ./confusion_matrices/ymja_rgb_finetuned.png > ./work_dirs/two_stream/ymja_rgb_finetune.txt
