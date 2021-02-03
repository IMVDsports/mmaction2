import torch

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'checkpoints/tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth'

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)

# test rawframe directory of a single video and show the result:
# video = 'data/ucf101/rawframes/ThrowDiscus/v_ThrowDiscus_g17_c05'
video = 'data/ucf101/videos/ThrowDiscus/v_ThrowDiscus_g17_c05.avi'

labels = 'data/ucf101/annotations/ucf101_labels.txt'
# results = inference_recognizer(model, video, labels, use_frames=True) # TODO Change te inference function to retrieve the number of frames from th eannotation file (not counting the total frames correctly)
results = inference_recognizer(model, video, labels) # TODO Does't work also with rawframes

# show the results
print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])