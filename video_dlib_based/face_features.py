# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:52:56 2019

@author: Muhammad Shahbaz (MS-IT-507815)
"""
from imutils.video import FPS
from imutils.face_utils import shape_to_np, rect_to_bb
from imutils import resize as imresize
import dlib
import cv2
from statistics import mean, stdev

# it reads a video file and return feature vector in numpy array
def extract_face_features(video_path, shape_predictor_path = 'shape_predictor_68_face_landmarks.dat', image_res_to_use = 400, show_video = False):
    # check
    invalid_video = False

    # loading feature extractors
    face_detector = dlib.get_frontal_face_detector()
    face_points_predictor =  dlib.shape_predictor(shape_predictor_path)

    # starting video stream
    video_stream = cv2.VideoCapture(video_path)

    # frames
    fps = FPS().start()

    # result array
    result_array = []

    # debug data
    frame_count = 0

    # extraction process
    while True:
        # reading a frame
        (grabbed, image) = video_stream.read()

        # invalid video
        if image is None:
            break

        # debug data
        frame_count = frame_count + 1

        # pre-processing a frame
        image = imresize(image, width=image_res_to_use)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detecting face
        detected_faces = face_detector(image, 0)

        # checking for more than once faces
        if len(detected_faces) > 1:
            invalid_video = True
            print(video_path, '-- multiple faces detected, aborting...')
            break

        # if no faces detected
        if len(detected_faces) == 0:
            result_array.append([0 for i in range(68*2)])
            if show_video:
                cv2.imshow('image_frame', image)

            key = cv2.waitKey(100) & 0xFF

        	# if the `q` key was pressed, break from the loop
            if key == ord('q'):
                break

            fps.update()
            continue

        # extracting face features from frame
        frame_feature_vector = []

        face_rect = detected_faces[0]
        # getting 68 feature points
        (X, Y, W, H) = rect_to_bb(face_rect)
        cv2.rectangle(image, (X, Y), (X+W, Y+H), (0,255,0), 1)
        face_points = face_points_predictor(image, face_rect)
        face_points = list(shape_to_np(face_points))

        # storing feature points to array
        for i, (x,y) in enumerate(face_points):
            frame_feature_vector.extend((x,y))
            if show_video:
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        if show_video:
            cv2.imshow('image_frame', image)

        key = cv2.waitKey(10) & 0xFF

    	# if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break

        fps.update()

        # putting to result array
        result_array.append(frame_feature_vector)

    # stopping video stream
    cv2.destroyAllWindows()
    fps.stop()
    video_stream.release()

    if not invalid_video and frame_count > 1:
        # pre-processing - top down fix
        previous_valid_frame = None
        for index, frame_feature_vector in enumerate(result_array):
            if not all(i==0 for i in frame_feature_vector): previous_valid_frame = frame_feature_vector
            if previous_valid_frame and all(i==0 for i in frame_feature_vector): result_array[index] = previous_valid_frame

        # reversing
        result_array.reverse()

        # pre-processing - bottom top fix
        previous_valid_frame = None
        for index, frame_feature_vector in enumerate(result_array):
            if not all(i==0 for i in frame_feature_vector): previous_valid_frame = frame_feature_vector
            if previous_valid_frame and all(i==0 for i in frame_feature_vector): result_array[index] = previous_valid_frame

        # reversing
        result_array.reverse()

        # calculating means and standard deviations
        means = []
        stdevs = []
        for i in range(68):
            feature = [x[i] for x in result_array]
            means.append(mean(feature))
            stdevs.append(stdev(feature))

        # raveling frames
        final_feature_vector = []
        final_feature_vector.extend(means)
        final_feature_vector.extend(stdevs)

        # debug data
        print(video_path, 'frames:', frame_count, '-- done.')

        # return a 136 dimensions data vector
        return final_feature_vector

    # else
    return None

if __name__ == '__main__':
    import time
    start = time.time()
    feature_vector = extract_face_features(video_path='train_data/trial_lie_001_000.mp4', image_res_to_use=400, show_video=True)
    end = time.time()
    print ('time taken:', end-start)