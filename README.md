# Py Lie Detect
An attempt (though totally not a successful one) to predict truth or lie in a video or audio using machine learning. It is still under progress, we are working on better feature encoding and evaluation techniques to train classifier.

# Usage:
There are two folders one for audio based (MFCC features) attemp and the other is video based (DLIB 68 point features) attempt. Both folders have a python scripts files "custom_data_evaluator.py" and "custom_test_score.py" respectively.
Those two scripts use the already trained classifier for video and audio features to predict answer for same test videos located in "test_data" folder. Execute these two files to generate results we got.

# Possible problems of these attempts:
1 - The data set is too much noisy.
2 - Video based features are not normalized for face scales and face orientation in video frames.
3 - Variable length data.
4 - Use of various sensors to capture videos and audios.
5 - Feature encoding used is wrong. (means and standard deviation of all frames of a video/audio are taken as one feature vector of that video)
