#!/usr/bin/env bash
python ./tools/test.py ./work_dirs/i3d_r50_100e_ymja_flow.py/i3d_r50_100e_ymja_flow.py ./work_dirs/i3d_r50_100e_ymja_flow.py/epoch_20.pth --out ./work_dirs/two_stream/ymja_flow.json
python ./tools/test.py ./work_dirs/i3d_r50_100e_ymja_rgb.py/i3d_r50_100e_ymja_rgb.py ./work_dirs/i3d_r50_100e_ymja_rgb.py/epoch_20.pth --out ./work_dirs/two_stream/ymja_rgb.json
python ./tools/test.py ./work_dirs/i3d_r50_100e_ymja_rgb_finetuned.py/i3d_r50_100e_ymja_rgb_finetuned.py ./work_dirs/i3d_r50_100e_ymja_rgb.py/epoch_5.pth --out ./work_dirs/two_stream/ymja_rgb_finetuned.json
python ./tools/test.py ./work_dirs/i3d_r50_100e_ymja_flow_finetuned.py/i3d_r50_100e_ymja_flow_finetuned.py ./work_dirs/i3d_r50_100e_ymja_flow.py/latest.pth --out ./work_dirs/two_stream/ymja_flow_finetuned.json

