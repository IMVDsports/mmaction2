#! /bin/bash

# Default configurations for TSN UCF101 pretrained model
# TODO Change test ann file for rgb to match the flow one so that the two steams results are the same
# python tools/test.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb/tsn_r50_1x1x3_50e_kinetics400_rgb_pretrained.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb/latest.pth --eval top_k_accuracy --out kinetics1_latest.json > kinetics1_latest.txt
# python tools/test.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb_lr=0.000625/tsn_r50_1x1x3_50e_kinetics400_rgb_pretrained.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb_lr=0.000625/epoch_15.pth --eval top_k_accuracy --out kinetics625_15.json > kinetics625_15.txt
# python tools/test.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb_lr=0.000625/tsn_r50_1x1x3_50e_kinetics400_rgb_pretrained.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_rgb_lr=0.000625/latest.pth --eval top_k_accuracy --out kinetics625_latest.json > kinetics625_latest.txt
# python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_flow.py --validate --seed 0 --deterministic
# python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_50e_kinetics400_flow.py --validate --seed 0 --deterministic

# Regular flow files
# python tools/test.py work_dirs/tsn_r50_1x1x3_2e_ucf101_flow/tsn_r50_1x1x3_75e_ucf101_flow.py work_dirs/tsn_r50_1x1x3_2e_ucf101_flow/epoch_60.pth --eval top_k_accuracy --out tsn_r50_1x1x3_75e_ucf101_flow_epoch_60.json > tsn_r50_1x1x3_75e_ucf101_flow_epoch_60.txt
# python tools/test.py work_dirs/tsn_r50_1x1x3_2e_ucf101_flow/tsn_r50_1x1x3_75e_ucf101_flow.py work_dirs/tsn_r50_1x1x3_2e_ucf101_flow/latest.pth --eval top_k_accuracy --out tsn_r50_1x1x3_75e_ucf101_flow_latest.json > tsn_r50_1x1x3_75e_ucf101_flow_latest.txt

# # Kinetics pre-trained flow files
# python tools/test.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_flow/tsn_r50_1x1x3_50e_kinetics400_flow.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_flow/epoch_45.pth --eval top_k_accuracy --out tsn_r50_1x1x3_50e_kinetics400_flow_epoch_45.json > tsn_r50_1x1x3_50e_kinetics400_flow_epoch_45.txt
# python tools/test.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_flow/tsn_r50_1x1x3_50e_kinetics400_flow.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_flow/latest.pth --eval top_k_accuracy --out tsn_r50_1x1x3_50e_kinetics400_flow_latest.json > tsn_r50_1x1x3_50e_kinetics400_flow_latest.txt

# Train RGB model on reduced dataset (to match flow predictions)
# python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py --validate --seed 0 --deterministic
# python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_50e_kinetics400_rgb_pretrained.py --validate --seed 0 --deterministic

# python tools/test.py work_dirs/tsn_r50_1x1x3_75e_ucf101_rgb/tsn_r50_1x1x3_75e_ucf101_rgb.py work_dirs/tsn_r50_1x1x3_75e_ucf101_rgb/epoch_20.pth --eval top_k_accuracy --out tsn_r50_1x1x3_75e_ucf101_rgb_epoch_20.json > tsn_r50_1x1x3_75e_ucf101_rgb_epoch_20.txt
# python tools/test.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_pretrained_rgb/tsn_r50_1x1x3_50e_kinetics400_rgb_pretrained.py work_dirs/tsn_r50_1x1x3_50e_kinetics400_pretrained_rgb/epoch_15.pth --eval top_k_accuracy --out tsn_r50_1x1x3_50e_kinetics400_pretrained_rgb_epoch_15.json > tsn_r50_1x1x3_50e_kinetics400_pretrained_rgb_epoch_15.txt

# Run on new dataset
# python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_75e_ymja_rgb.py --validate --seed 0 --deterministic
# python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_75e_ymja_flow.py --validate --seed 0 --deterministic
# python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_50e_kinetics400_rgb_ymja.py --validate --seed 0 --deterministic
# python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_50e_kinetics400_flow_ymja.py --validate --seed 0 --deterministic

# Test new dataset
# python tools/test.py work_dirs/YMJA/tsn_r50_1x1x3_30e_flow/tsn_r50_1x1x3_75e_ymja_flow.py work_dirs/YMJA/tsn_r50_1x1x3_30e_flow/epoch_20.pth --eval top_k_accuracy --out final_flow_5classes.json > final_flow_5classes.txt
# python tools/test.py work_dirs/YMJA/tsn_r50_1x1x3_30e_rgb/tsn_r50_1x1x3_75e_ymja_rgb.py work_dirs/YMJA/tsn_r50_1x1x3_30e_rgb/epoch_30.pth --eval top_k_accuracy --out final_rgb_5classes.json > final_rgb_5classes.txt
# python tools/test.py work_dirs/YMJA/tsn_r50_1x1x3_40e_kinetics400_flow/tsn_r50_1x1x3_50e_kinetics400_flow_ymja.py work_dirs/YMJA/tsn_r50_1x1x3_40e_kinetics400_flow/epoch_25.pth --eval top_k_accuracy --out final_flow_pretrained_5classes.json > final_flow_pretrained_5classes.txt
# python tools/test.py work_dirs/YMJA/tsn_r50_1x1x3_40e_kinetics400_pretrained_rgb/tsn_r50_1x1x3_50e_kinetics400_rgb_ymja.py work_dirs/YMJA/tsn_r50_1x1x3_40e_kinetics400_pretrained_rgb/epoch_5.pth --eval top_k_accuracy --out final_rgb_pretrained_5classes.json > final_rgb_pretrained_5classes.txt

# Fusion
# python fusion.py 2 final_rgb_5classes.json final_flow_5classes.json > fusion_reg_1_1.txt 
# python fusion.py 2 final_rgb_5classes.json final_flow_5classes.json --weights 1 0.5 > fusion_reg_1_05.txt 
# python fusion.py 2 final_rgb_5classes.json final_flow_5classes.json --weights 0.5 1 > fusion_reg_05_1.txt

# python fusion.py 2 final_rgb_pretrained_5classes.json final_flow_pretrained_5classes.json > fusion_pretrained_1_1.txt 
# python fusion.py 2 final_rgb_pretrained_5classes.json final_flow_pretrained_5classes.json --weights 1 0.5 > fusion_pretrained_1_05.txt 
# python fusion.py 2 final_rgb_pretrained_5classes.json final_flow_pretrained_5classes.json --weights 0.5 1 > fusion_pretrained_05_1.txt

# Run randomly 3 times
# for i in {1..3}
#     do
#         python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_75e_ymja_rgb.py --validate --work-dir "./work_dirs/YMJA/rand/tsn_rgb_${i}/"
#         python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_75e_ymja_flow.py --validate --work-dir "./work_dirs/YMJA/rand/tsn_flow_${i}/"
#         python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_50e_kinetics400_rgb_ymja.py --validate --work-dir "./work_dirs/YMJA/rand/tsn_kinetics400_rgb_${i}/"
#         python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_50e_kinetics400_flow_ymja.py --validate --work-dir "./work_dirs/YMJA/rand/tsn_kinetics400_flow_${i}/"
#     done

# # Test
TEST_PATH=work_dirs/YMJA/rand
# Flow
python tools/test.py $TEST_PATH/tsn_flow/tsn_r50_1x1x3_75e_ymja_flow.py $TEST_PATH/tsn_flow/epoch_25.pth --eval top_k_accuracy --out results/tsn_flow/potion_run0.json
python tools/test.py $TEST_PATH/tsn_flow_1/tsn_r50_1x1x3_75e_ymja_flow.py $TEST_PATH/tsn_flow_1/epoch_20.pth --eval top_k_accuracy --out results/tsn_flow/potion_run1.json
python tools/test.py $TEST_PATH/tsn_flow_2/tsn_r50_1x1x3_75e_ymja_flow.py $TEST_PATH/tsn_flow_2/epoch_25.pth --eval top_k_accuracy --out results/tsn_flow/potion_run2.json
python tools/test.py $TEST_PATH/tsn_flow_3/tsn_r50_1x1x3_75e_ymja_flow.py $TEST_PATH/tsn_flow_3/epoch_25.pth --eval top_k_accuracy --out results/tsn_flow/potion_run3.json

# RGB
python tools/test.py $TEST_PATH/tsn_rgb/tsn_r50_1x1x3_75e_ymja_rgb.py $TEST_PATH/tsn_rgb/epoch_25.pth --eval top_k_accuracy --out results/tsn_rgb/potion_run0.json
python tools/test.py $TEST_PATH/tsn_rgb_1/tsn_r50_1x1x3_75e_ymja_rgb.py $TEST_PATH/tsn_rgb_1/epoch_15.pth --eval top_k_accuracy --out results/tsn_rgb/potion_run1.json
python tools/test.py $TEST_PATH/tsn_rgb_2/tsn_r50_1x1x3_75e_ymja_rgb.py $TEST_PATH/tsn_rgb_2/epoch_30.pth --eval top_k_accuracy --out results/tsn_rgb/potion_run2.json
python tools/test.py $TEST_PATH/tsn_rgb_3/tsn_r50_1x1x3_75e_ymja_rgb.py $TEST_PATH/tsn_rgb_3/epoch_30.pth --eval top_k_accuracy --out results/tsn_rgb/potion_run3.json

# Pretrained Flow
python tools/test.py $TEST_PATH/tsn_kinetics400_flow/tsn_r50_1x1x3_50e_kinetics400_flow_ymja.py $TEST_PATH/tsn_kinetics400_flow/epoch_20.pth --eval top_k_accuracy --out results/tsn_pretrained_flow/potion_run0.json
python tools/test.py $TEST_PATH/tsn_kinetics400_flow_1/tsn_r50_1x1x3_50e_kinetics400_flow_ymja.py $TEST_PATH/tsn_kinetics400_flow_1/epoch_15.pth --eval top_k_accuracy --out results/tsn_pretrained_flow/potion_run1.json
python tools/test.py $TEST_PATH/tsn_kinetics400_flow_2/tsn_r50_1x1x3_50e_kinetics400_flow_ymja.py $TEST_PATH/tsn_kinetics400_flow_2/epoch_10.pth --eval top_k_accuracy --out results/tsn_pretrained_flow/potion_run2.json
python tools/test.py $TEST_PATH/tsn_kinetics400_flow_3/tsn_r50_1x1x3_50e_kinetics400_flow_ymja.py $TEST_PATH/tsn_kinetics400_flow_3/epoch_20.pth --eval top_k_accuracy --out results/tsn_pretrained_flow/potion_run3.json

# Pretrained RGB
python tools/test.py $TEST_PATH/tsn_kinetics400_rgb/tsn_r50_1x1x3_50e_kinetics400_rgb_ymja.py $TEST_PATH/tsn_kinetics400_rgb/epoch_30.pth --eval top_k_accuracy --out results/tsn_pretrained_rgb/potion_run0.json
python tools/test.py $TEST_PATH/tsn_kinetics400_rgb_1/tsn_r50_1x1x3_50e_kinetics400_rgb_ymja.py $TEST_PATH/tsn_kinetics400_rgb_1/epoch_10.pth --eval top_k_accuracy --out results/tsn_pretrained_rgb/potion_run1.json
python tools/test.py $TEST_PATH/tsn_kinetics400_rgb_2/tsn_r50_1x1x3_50e_kinetics400_rgb_ymja.py $TEST_PATH/tsn_kinetics400_rgb_2/epoch_15.pth --eval top_k_accuracy --out results/tsn_pretrained_rgb/potion_run2.json
python tools/test.py $TEST_PATH/tsn_kinetics400_rgb_3/tsn_r50_1x1x3_50e_kinetics400_rgb_ymja.py $TEST_PATH/tsn_kinetics400_rgb_3/epoch_20.pth --eval top_k_accuracy --out results/tsn_pretrained_rgb/potion_run3.json

# Test all results
# Flow (classic)
# python fusion.py 1 results/tsn_flow/run0.json > results/tsn_flow/output/run0.txt 
# python fusion.py 1 results/tsn_flow/run1.json > results/tsn_flow/output/run1.txt 
# python fusion.py 1 results/tsn_flow/run2.json > results/tsn_flow/output/run2.txt
# python fusion.py 1 results/tsn_flow/run3.json > results/tsn_flow/output/run3.txt # this

# python fusion.py 1 results/tsn_rgb/run0.json > results/tsn_rgb/output/run0.txt 
# python fusion.py 1 results/tsn_rgb/run1.json > results/tsn_rgb/output/run1.txt 
# python fusion.py 1 results/tsn_rgb/run2.json > results/tsn_rgb/output/run2.txt
# python fusion.py 1 results/tsn_rgb/run3.json > results/tsn_rgb/output/run3.txt # this

# python fusion.py 1 results/tsn_pretrained_flow/run0.json > results/tsn_pretrained_flow/output/run0.txt 
# python fusion.py 1 results/tsn_pretrained_flow/run1.json > results/tsn_pretrained_flow/output/run1.txt 
# python fusion.py 1 results/tsn_pretrained_flow/run2.json > results/tsn_pretrained_flow/output/run2.txt
# python fusion.py 1 results/tsn_pretrained_flow/run3.json > results/tsn_pretrained_flow/output/run3.txt # this

# python fusion.py 1 results/tsn_pretrained_rgb/run0.json > results/tsn_pretrained_rgb/output/run0.txt 
# python fusion.py 1 results/tsn_pretrained_rgb/run1.json > results/tsn_pretrained_rgb/output/run1.txt 
# python fusion.py 1 results/tsn_pretrained_rgb/run2.json > results/tsn_pretrained_rgb/output/run2.txt # this
# python fusion.py 1 results/tsn_pretrained_rgb/run3.json > results/tsn_pretrained_rgb/output/run3.txt

# # Best fusion
# python fusion.py 2 results/tsn_pretrained_rgb/run2.json results/tsn_pretrained_flow/run3.json --weights 1 0.5 > poster_value_vf.txt 

# # Avg (std) fusion
# python fusion.py 2 results/tsn_pretrained_rgb/run0.json results/tsn_pretrained_flow/run0.json --weights 1 0.5 > poster_value_0.txt 
# python fusion.py 2 results/tsn_pretrained_rgb/run1.json results/tsn_pretrained_flow/run1.json --weights 1 0.5 > poster_value_1.txt 
# python fusion.py 2 results/tsn_pretrained_rgb/run2.json results/tsn_pretrained_flow/run2.json --weights 1 0.5 > poster_value_2.txt 
# python fusion.py 2 results/tsn_pretrained_rgb/run3.json results/tsn_pretrained_flow/run3.json --weights 1 0.5 > poster_value_3.txt 

# Change the weights
# python fusion.py 2 results/tsn_pretrained_rgb/run0.json results/tsn_pretrained_flow/run0.json > results/fusion/pretrained_streams/run0_1_1.txt
# python fusion.py 2 results/tsn_pretrained_rgb/run1.json results/tsn_pretrained_flow/run1.json > results/fusion/pretrained_streams/run1_1_1.txt
# python fusion.py 2 results/tsn_pretrained_rgb/run2.json results/tsn_pretrained_flow/run2.json > results/fusion/pretrained_streams/run2_1_1.txt
# python fusion.py 2 results/tsn_pretrained_rgb/run3.json results/tsn_pretrained_flow/run3.json > results/fusion/pretrained_streams/run3_1_1.txt

# python fusion.py 2 results/tsn_pretrained_rgb/run0.json results/tsn_pretrained_flow/run0.json --weights 0.5 1 > results/fusion/pretrained_streams/run0_05_1.txt
# python fusion.py 2 results/tsn_pretrained_rgb/run1.json results/tsn_pretrained_flow/run1.json --weights 0.5 1 > results/fusion/pretrained_streams/run1_05_1.txt
# python fusion.py 2 results/tsn_pretrained_rgb/run2.json results/tsn_pretrained_flow/run2.json --weights 0.5 1 > results/fusion/pretrained_streams/run2_05_1.txt
# python fusion.py 2 results/tsn_pretrained_rgb/run3.json results/tsn_pretrained_flow/run3.json --weights 0.5 1 > results/fusion/pretrained_streams/run3_05_1.txt

# python fusion.py 2 results/tsn_rgb/run0.json results/tsn_flow/run0.json --weights 1 0.5 > results/fusion/reg_streams/run0_1_05.txt
# python fusion.py 2 results/tsn_rgb/run1.json results/tsn_flow/run1.json --weights 1 0.5 > results/fusion/reg_streams/run1_1_05.txt
# python fusion.py 2 results/tsn_rgb/run2.json results/tsn_flow/run2.json --weights 1 0.5 > results/fusion/reg_streams/run2_1_05.txt
# python fusion.py 2 results/tsn_rgb/run3.json results/tsn_flow/run3.json --weights 1 0.5 > results/fusion/reg_streams/run3_1_05.txt

# python fusion.py 2 results/tsn_rgb/run0.json results/tsn_flow/run0.json --weights 0.5 1 > results/fusion/reg_streams/run0_05_1.txt
# python fusion.py 2 results/tsn_rgb/run1.json results/tsn_flow/run1.json --weights 0.5 1 > results/fusion/reg_streams/run1_05_1.txt
# python fusion.py 2 results/tsn_rgb/run2.json results/tsn_flow/run2.json --weights 0.5 1 > results/fusion/reg_streams/run2_05_1.txt
# python fusion.py 2 results/tsn_rgb/run3.json results/tsn_flow/run3.json --weights 0.5 1 > results/fusion/reg_streams/run3_05_1.txt

# python fusion.py 2 results/tsn_rgb/run0.json results/tsn_flow/run0.json > results/fusion/reg_streams/run0_1_1.txt
# python fusion.py 2 results/tsn_rgb/run1.json results/tsn_flow/run1.json > results/fusion/reg_streams/run1_1_1.txt
# python fusion.py 2 results/tsn_rgb/run2.json results/tsn_flow/run2.json > results/fusion/reg_streams/run2_1_1.txt
# python fusion.py 2 results/tsn_rgb/run3.json results/tsn_flow/run3.json > results/fusion/reg_streams/run3_1_1.txt

# Potion results on Ben's dataset
# TEST_PATH=work_dirs/YMJA/rand
# python tools/test.py $TEST_PATH/tsn_flow_3/tsn_r50_1x1x3_75e_ymja_flow.py $TEST_PATH/tsn_flow_3/epoch_25.pth --eval top_k_accuracy --out results/tsn_flow/potion.json
# python tools/test.py $TEST_PATH/tsn_rgb_3/tsn_r50_1x1x3_75e_ymja_rgb.py $TEST_PATH/tsn_rgb_3/epoch_30.pth --eval top_k_accuracy --out results/tsn_rgb/potion.json
# python tools/test.py $TEST_PATH/tsn_kinetics400_flow_3/tsn_r50_1x1x3_50e_kinetics400_flow_ymja.py $TEST_PATH/tsn_kinetics400_flow_3/epoch_20.pth --eval top_k_accuracy --out results/tsn_pretrained_flow/potion.json
# python tools/test.py $TEST_PATH/tsn_kinetics400_rgb_2/tsn_r50_1x1x3_50e_kinetics400_rgb_ymja.py $TEST_PATH/tsn_kinetics400_rgb_2/epoch_15.pth --eval top_k_accuracy --out results/tsn_pretrained_rgb/potion.json

# Potion with pretrained
POTION_FILE='PoTion_predictions_before_softmax.json'
python fusion.py 3 results/tsn_pretrained_rgb/potion_run0.json results/tsn_pretrained_flow/potion_run0.json $POTION_FILE > results/fusion/potion_pre/run0_1_1_1.txt
python fusion.py 3 results/tsn_pretrained_rgb/potion_run1.json results/tsn_pretrained_flow/potion_run1.json $POTION_FILE > results/fusion/potion_pre/run1_1_1_1.txt
python fusion.py 3 results/tsn_pretrained_rgb/potion_run2.json results/tsn_pretrained_flow/potion_run2.json $POTION_FILE > results/fusion/potion_pre/run2_1_1_1.txt
python fusion.py 3 results/tsn_pretrained_rgb/potion_run3.json results/tsn_pretrained_flow/potion_run3.json $POTION_FILE > results/fusion/potion_pre/run3_1_1_1.txt

python fusion.py 3 results/tsn_pretrained_rgb/potion_run0.json results/tsn_pretrained_flow/potion_run0.json $POTION_FILE --weights 1 0.5 1.5 > results/fusion/potion_pre/run0_1_05_15.txt 
python fusion.py 3 results/tsn_pretrained_rgb/potion_run1.json results/tsn_pretrained_flow/potion_run1.json $POTION_FILE --weights 1 0.5 1.5 > results/fusion/potion_pre/run1_1_05_15.txt 
python fusion.py 3 results/tsn_pretrained_rgb/potion_run2.json results/tsn_pretrained_flow/potion_run2.json $POTION_FILE --weights 1 0.5 1.5 > results/fusion/potion_pre/run2_1_05_15.txt 
python fusion.py 3 results/tsn_pretrained_rgb/potion_run3.json results/tsn_pretrained_flow/potion_run3.json $POTION_FILE --weights 1 0.5 1.5 > results/fusion/potion_pre/run3_1_05_15.txt

# Potion with regular
python fusion.py 3 results/tsn_rgb/potion_run0.json results/tsn_flow/potion_run0.json $POTION_FILE > results/fusion/potion_reg/run0_1_1_1.txt
python fusion.py 3 results/tsn_rgb/potion_run1.json results/tsn_flow/potion_run1.json $POTION_FILE > results/fusion/potion_reg/run1_1_1_1.txt
python fusion.py 3 results/tsn_rgb/potion_run2.json results/tsn_flow/potion_run2.json $POTION_FILE > results/fusion/potion_reg/run2_1_1_1.txt
python fusion.py 3 results/tsn_rgb/potion_run3.json results/tsn_flow/potion_run3.json $POTION_FILE > results/fusion/potion_reg/run3_1_1_1.txt

python fusion.py 3 results/tsn_rgb/potion_run0.json results/tsn_flow/potion_run0.json $POTION_FILE --weights 1 0.5 1.5 > results/fusion/potion_reg/run0_1_05_15.txt
python fusion.py 3 results/tsn_rgb/potion_run1.json results/tsn_flow/potion_run1.json $POTION_FILE --weights 1 0.5 1.5 > results/fusion/potion_reg/run1_1_05_15.txt
python fusion.py 3 results/tsn_rgb/potion_run2.json results/tsn_flow/potion_run2.json $POTION_FILE --weights 1 0.5 1.5 > results/fusion/potion_reg/run2_1_05_15.txt
python fusion.py 3 results/tsn_rgb/potion_run3.json results/tsn_flow/potion_run3.json $POTION_FILE --weights 1 0.5 1.5 > results/fusion/potion_reg/run3_1_05_15.txt