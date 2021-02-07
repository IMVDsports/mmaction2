# Default configurations for TSN UCF101 pretrained model
# TODO Change test ann file for rgb to match the flow one so that the two steams results are the same
# python tools/test.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb/tsn_r50_1x1x3_50e_kinetics400_rgb_pretrained.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb/latest.pth --eval top_k_accuracy --out kinetics1_latest.json > kinetics1_latest.txt
# python tools/test.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb_lr=0.000625/tsn_r50_1x1x3_50e_kinetics400_rgb_pretrained.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb_lr=0.000625/epoch_15.pth --eval top_k_accuracy --out kinetics625_15.json > kinetics625_15.txt
# python tools/test.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb_lr=0.000625/tsn_r50_1x1x3_50e_kinetics400_rgb_pretrained.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb_lr=0.000625/latest.pth --eval top_k_accuracy --out kinetics625_latest.json > kinetics625_latest.txt
# python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_flow.py --validate --seed 0 --deterministic
# python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_50e_kinetics400_flow.py --validate --seed 0 --deterministic

# Regular flow files
# python tools/test.py work_dirs/tsn_r50_1x1x3_2e_ucf101_flow/tsn_r50_1x1x3_75e_ucf101_flow.py work_dirs/tsn_r50_1x1x3_2e_ucf101_flow/epoch_60.pth --eval top_k_accuracy --out tsn_r50_1x1x3_75e_ucf101_flow_epoch_60.json > tsn_r50_1x1x3_75e_ucf101_flow_epoch_60.txt
python tools/test.py work_dirs/tsn_r50_1x1x3_2e_ucf101_flow/tsn_r50_1x1x3_75e_ucf101_flow.py work_dirs/tsn_r50_1x1x3_2e_ucf101_flow/latest.pth --eval top_k_accuracy --out tsn_r50_1x1x3_75e_ucf101_flow_latest.json > tsn_r50_1x1x3_75e_ucf101_flow_latest.txt

# Kinetics pre-trained flow files
python tools/test.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_flow/tsn_r50_1x1x3_50e_kinetics400_flow.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_flow/epoch_45.pth --eval top_k_accuracy --out tsn_r50_1x1x3_50e_kinetics400_flow_epoch_45.json > tsn_r50_1x1x3_50e_kinetics400_flow_epoch_45.txt
python tools/test.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_flow/tsn_r50_1x1x3_50e_kinetics400_flow.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_flow/latest.pth --eval top_k_accuracy --out tsn_r50_1x1x3_50e_kinetics400_flow_latest.json > tsn_r50_1x1x3_50e_kinetics400_flow_latest.txt