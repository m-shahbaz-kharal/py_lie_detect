# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 02:30:34 2019

@author: Muhammad Shahbaz (MS-18-IT-507815)
"""
from face_features import extract_face_features
from glob import glob
from joblib import dump, load
import numpy as np

def build_csv(training_videos_dir, video_list_file):
    # invalid data
    invalid_files = []

    # data set vars
    train_data = None

    # reading video list
    training_set_dirs = None
    with open(video_list_file, 'r') as video_list_txt: training_set_dirs = video_list_txt.readlines()
    training_set_dirs[-1] += '\n'

    # printing debug data
    print('total training videos:', len(training_set_dirs))

    for index, video_dir in enumerate(training_set_dirs):
        video_dir = video_dir.replace('.mp4\n', '')

        # printing progress
        print('[%d] extracting features from: %s' % (index, video_dir))

        # valid data check boolean
        for index, video_path in enumerate(glob(training_videos_dir+'\\'+video_dir+'_*.mp4')):
            Xcurrent = extract_face_features(video_path=video_path)

            # validity check
            if Xcurrent and any(Xcurrent):
                if video_dir.find('lie') >= 0:
                    Xcurrent.append(1)
                    if train_data is None:
                        train_data = Xcurrent
                    else:
                        train_data = np.vstack((train_data, Xcurrent))
                else:
                    Xcurrent.append(-1)
                    if train_data is None:
                        train_data = Xcurrent
                    else:
                        train_data = np.vstack((train_data, Xcurrent))
            else:
                invalid_files.append(video_dir)

    # dump
    dump(train_data, 'out/train_data.dump')

    print('saving invalid files...')
    # saving invalid files
    if len(invalid_files) > 0:
        invalid_files_txt = open('out/invalid_files.txt', 'w+')
        for f in invalid_files: invalid_files_txt.write('file: %s\r\n' % f)
        invalid_files_txt.close()

    print('Writing CSV...')
    # exporting to csv
    np.savetxt('out/train_data.csv', train_data, delimiter=',')

    return train_data

if __name__ == '__main__':
    import time
    start = time.time()
    train_data = build_csv('train_data', 'train_video_list.txt')
    end = time.time()
    print('Total time:', end-start)