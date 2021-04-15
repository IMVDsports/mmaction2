import os
import math
import random
from collections import defaultdict

def main():
    path_to_classnames = os.path.join('data', 'YMJA', 'rawframes_sampled')
    classNames = os.listdir(path_to_classnames)
    class_labels = {
        'No_penalty': 0,
        'Tripping': 1,
        'Hooking': 2,
        'Slashing': 3,
        'Holding': 4
    }
    video_names = []
    # Iterate through the folders in YMJA
    for class_name in classNames:
        tmp_path = os.path.join(path_to_classnames, class_name)
        # Store penalty video name, 64, the folder name (class) 
        for video in os.listdir(tmp_path):
            video_names.append((os.path.join(class_name, video.split('.')[0]), 64, class_labels[class_name]))

    # Take a random permutation to create a train.txt, val.txt (which is used for testing)
    random.seed(0) # for consistency
    random.shuffle(video_names)
    train_ratio = 0.7
    n_train_videos = math.ceil(train_ratio * len(video_names))
    val_ratio = 0.15
    n_val_videos = math.floor(val_ratio * len(video_names))
    #  test videos is the rest
    
    with open(os.path.join('data', 'YMJA', 'ymja_train_split_1_rawframes.txt'), 'w') as f:
        for line in video_names[:n_train_videos]:
            name, num_frames, label = line 
            f.write('{} {} {}\n'.format(name, num_frames, label))

    with open(os.path.join('data', 'YMJA', 'ymja_val_split_1_rawframes.txt'), 'w') as f:
        for line in video_names[n_train_videos:n_train_videos+n_val_videos]:
            name, num_frames, label = line 
            f.write('{} {} {}\n'.format(name, num_frames, label))

    with open(os.path.join('data', 'YMJA', 'ymja_test_split_1_rawframes.txt'), 'w') as f:
        for line in video_names[n_train_videos+n_val_videos:]:
            name, num_frames, label = line 
            f.write('{} {} {}\n'.format(name, num_frames, label))


if __name__ == '__main__':
    main()