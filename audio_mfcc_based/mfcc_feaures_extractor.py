# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:07:31 2019

@author: Muhammad Shahbaz (MS-18-IT-507815)
"""
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from joblib import dump
import os
from statistics import mean, stdev

def get_audio_from_video(video_path, out_folder):
    import subprocess

    out_file_name = os.path.basename(video_path).split('.')[0]
    subprocess.call(['ffmpeg', '-i', video_path, '-ac', '1', out_folder+'/'+out_file_name+'.wav'])

    return out_folder+'/'+out_file_name+'.wav'

def extract_mfcc_features(audio_path):
    (rate, sig) = wav.read(audio_path)
    mfcc_features = mfcc(sig, rate, nfft=1200)
    return mfcc_features

def create_dataset_using_full_clips(data_path, out_folder, dump_file):
    from os import listdir

    result = None
    sample = []
    last_folder = ''
    for folder in sorted(listdir(data_path), key=lambda item: (int(item.split('_')[1]))):
        if len(sample) > 0:
            mfcc_stack = None
            if not os.path.exists(out_folder+'/'+last_folder):
                os.makedirs(out_folder+'/'+last_folder+'/')
            for x in sample:
                print(x, last_folder)
                audio_file = get_audio_from_video(data_path+'/'+last_folder+'/'+x, out_folder+'/'+last_folder)
                mfcc_features = extract_mfcc_features(audio_file)
                if mfcc_stack is None:
                    mfcc_stack = mfcc_features
                else:
                    mfcc_stack = np.vstack((mfcc_stack, mfcc_features))
            means = []
            stdevs = []
            for i in range(13):
                feature = [x[i] for x in mfcc_stack]
                means.append(mean(feature))
                stdevs.append(stdev(feature))
            final_feature_vector = []
            final_feature_vector.extend(means)
            final_feature_vector.extend(stdevs)
            sid = int(last_folder.split('_')[1])
            fv = final_feature_vector
            lbl = 1 if last_folder[-3:] == 'lie' else -1
            if result is None:
                result = np.hstack((sid,fv,lbl))
            else:
                result = np.vstack((result,np.hstack((sid,fv,lbl))))
        sample = []
        last_folder = folder
        for index, file in enumerate(sorted(listdir(data_path+'/'+folder))):
            if file.endswith('000.mp4'):
                if len(sample) > 0:
                    mfcc_stack = None
                    if not os.path.exists(out_folder+'/'+last_folder):
                        os.makedirs(out_folder+'/'+last_folder+'/')
                    for x in sample:
                        print(x,folder)
                        audio_file = get_audio_from_video(data_path+'/'+folder+'/'+x, out_folder+'/'+last_folder)
                        mfcc_features = extract_mfcc_features(audio_file)
                        if mfcc_stack is None:
                            mfcc_stack = mfcc_features
                        else:
                            mfcc_stack = np.vstack((mfcc_stack, mfcc_features))
                    means = []
                    stdevs = []
                    for i in range(13):
                        feature = [x[i] for x in mfcc_stack]
                        means.append(mean(feature))
                        stdevs.append(stdev(feature))
                    final_feature_vector = []
                    final_feature_vector.extend(means)
                    final_feature_vector.extend(stdevs)
                    sid = int(folder.split('_')[1])
                    fv = final_feature_vector
                    lbl = 1 if folder[-3:] == 'lie' else -1
                    if result is None:
                        result = np.hstack((sid,fv,lbl))
                    else:
                        result = np.vstack((result,np.hstack((sid,fv,lbl))))
                sample = []
                sample.append(file)
            else:
                sample.append(file)
    if len(sample) > 0:
        mfcc_stack = None
        if not os.path.exists(out_folder+'/'+last_folder):
            os.makedirs(out_folder+'/'+last_folder+'/')
        for x in sample:
            print(x,last_folder)
            audio_file = get_audio_from_video(data_path+'/'+last_folder+'/'+x, out_folder+'/'+last_folder)
            mfcc_features = extract_mfcc_features(audio_file)
            if mfcc_stack is None:
                mfcc_stack = mfcc_features
            else:
                mfcc_stack = np.vstack((mfcc_stack, mfcc_features))
        means = []
        stdevs = []
        for i in range(13):
            feature = [x[i] for x in mfcc_stack]
            means.append(mean(feature))
            stdevs.append(stdev(feature))
        final_feature_vector = []
        final_feature_vector.extend(means)
        final_feature_vector.extend(stdevs)
        sid = int(last_folder.split('_')[1])
        fv = final_feature_vector
        lbl = 1 if last_folder[-3:] == 'lie' else -1
        if result is None:
            result = np.hstack((sid,fv,lbl))
        else:
            result = np.vstack((result,np.hstack((sid,fv,lbl))))

    dump(result, 'out/'+dump_file+'.dump')
    return result

def create_dataset_using_clip_chunks(data_path, out_folder, dump_file):
    result = None
    for folder in sorted(os.listdir(data_path), key=lambda item: (int(item.split('_')[1]))):
        for index, file in enumerate(sorted(os.listdir(data_path+'/'+folder))):
            if not os.path.exists(out_folder+'/'+folder):
                os.makedirs(out_folder+'/'+folder+'/')
            print(file, folder)
            audio_file = get_audio_from_video(data_path+'/'+folder+'/'+file, out_folder+'/'+folder)
            mfcc_features = extract_mfcc_features(audio_file)
            sid = int(folder.split('_')[1])
            fv = []
            for i in range(13):
                fv.append(mean(mfcc_features[:,i]))
                fv.append(stdev(mfcc_features[:,i]))
            lbl = 1 if folder[-3:] == 'lie' else -1
            if result is None:
                result = np.hstack((sid,fv,lbl))
            else:
                result = np.vstack((result,np.hstack((sid,fv,lbl))))

    dump(result, 'out/'+dump_file+'.dump')
    return result

if __name__ == "__main__":
    """
    option 1 => uses the full length audio training data
    option 2 => uses the individual clip (4 sec chunks of every audio) training data
    """

    option = 1

    features = None

    if option == 1:
        features = create_dataset_using_full_clips('train_data', 'temporary_audio_storage', 'full_length_data')
    else:
        features = create_dataset_using_clip_chunks('train_data', 'temporary_audio_storage', 'clip_based_data')