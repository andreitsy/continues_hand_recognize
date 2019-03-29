import sys
import cv2
import os
import pandas as pd
import numpy as np
from sys import platform
from argparse import ArgumentParser


# Import Openpose (Windows/Ubuntu/OSX) assumed that build with open pose is located in "build" directory
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
python_path = os.path.join(dir_path, "build", "python", "openpose", "Release")
path_release = os.path.join(dir_path, "build", "x64", "Release")
path_bin = os.path.join(dir_path, "build", "bin")
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(python_path)
        os.environ['PATH'] = "{};{};{}".format(os.environ['PATH'], path_release, path_bin)
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu),
        # you can also access the OpenPose/python module from there.
        # This will install OpenPose and the python library at your desired installation path.
        # Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` '
          'CMake and have this Python script in the right folder?')
    raise e


def initialize_open_pose_hand(model_folder=os.path.join(path_release, "models"), hand_detector=2):
    params = dict()
    params["model_folder"] = model_folder
    params["hand"] = True
    params["hand_detector"] = hand_detector
    params["body"] = 0

    # Starting OpenPose
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()
    return op_wrapper


def read_video(path_to_video_file):
    """
    open_cv can read the video in the different formats e.g. mp4, pov, avi...
    :param path_to_video_file: path to video file
    :return: open_cv object
    """
    cv_video = cv2.VideoCapture(path_to_video_file)
    assert cv_video.isOpened(), "Video cannot be opened!"
    return cv_video


def detect_key_points(video_to_process, op_wrapper):
    """
    Creates rectangle and process it
    :param op_wrapper: open pose wrapper object
    :param video_to_process: open_cv object to process
    :return: np.array with data
    """
    width = video_to_process.get(3)  # float width of video
    height = video_to_process.get(4)  # float high of video

    # by default rectangle for right hand by center
    right_hand_rectangle = [
        # center
        [
            op.Rectangle(0., 0., 0., 0.),
            op.Rectangle(0.2 * width, 0.1 * height, height * 0.85, height * 0.85)
        ]
    ]

    datum = op.Datum()
    data = None
    timestamps = []

    while video_to_process.isOpened():
        # Capture frame-by-frame
        ret, frame = video_to_process.read()
        if ret:
            datum.cvInputData = frame
            datum.handRectangles = right_hand_rectangle
            op_wrapper.emplaceAndPop([datum])
            cv2.imshow("OpenPose 1.4.0. Right hand keypoints", datum.cvOutputData)
            timestamps.append(video_to_process.get(cv2.CAP_PROP_POS_MSEC))
            if data is not None:
                data = np.append(data, datum.handKeypoints[1].reshape(1, 63), axis=0)
            else:
                data = datum.handKeypoints[1].reshape(1, 63)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # add timestamps to beginning
    data_with_time = np.append(np.array(timestamps).reshape(len(timestamps), 1), data, axis=1)
    return data_with_time


def pretty_output(data, output):
    columns = ["time"]
    for i in range(21):
        columns += ["x" + str(i)]
        columns += ["y" + str(i)]
        columns += ["c" + str(i)]
    pandas_key_points = pd.DataFrame(data, columns=columns)
    pandas_key_points[columns].to_csv(output, encoding='utf-8', index=False)


def main(args):
    video_to_process = read_video(args.video_file)
    op_wrapper = initialize_open_pose_hand()
    data = detect_key_points(video_to_process, op_wrapper)
    pretty_output(data, args.output_file)

    # when everything done, release the video capture object
    video_to_process.release()
    # closes all the frames
    cv2.destroyAllWindows()


def parser_arguments():
    """
    Take path video as argument, detects key points via openpose for right hand and save output
    to file in the csv format with header:
    time,x0,y0,c0,...,x20,y20,c20

    :return: ArgumentParser
    """
    parser = ArgumentParser(description='Generate keypoints file for given video')
    parser.add_argument('-o', '--output-file', type=str, help='File name of output', default="output.csv")
    parser.add_argument('-v', '--video-file', type=str, help='Video file to process', required=True)
    return parser


if __name__ == '__main__':
    parser_cmd = parser_arguments()
    main(parser_cmd.parse_args())
