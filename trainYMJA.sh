#!/usr/bin/env bash
python ./tools/train.py ./configs/recognition/i3d/i3d_r50_100e_ymja_flow.py --validate --deterministic --seed 0
python ./tools/train.py ./configs/recognition/i3d/i3d_r50_100e_ymja_flow_finetuned.py --validate --deterministic --seed 0
python ./tools/train.py ./configs/recognition/i3d/i3d_r50_100e_ymja_rgb.py --validate --deterministic --seed 0
python ./tools/train.py ./configs/recognition/i3d/i3d_r50_100e_ymja_rgb_finetuned.py --validate --deterministic --seed 0