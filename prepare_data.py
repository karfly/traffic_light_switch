from __future__ import print_function, division

import os
from os.path import join as pj
import shutil
from glob import glob

import numpy as np
np.random.seed = 0  # for reproducibility

import argparse


from tqdm import tqdm

import cv2

from sklearn.model_selection import train_test_split


# Funtions
def list_dir_with_full_paths(dir_path):
    dir_abs_path = os.path.abspath(dir_path)
    return sorted([os.path.join(dir_abs_path, file_name) for file_name in os.listdir(dir_abs_path)])


def extract_images_from_video(video_path, images_dir, switch_frame=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    video_capture = cv2.VideoCapture(video_path)
    _, _ = video_capture.read()  # mock read

    count = 0
    success, image = video_capture.read()
    while success:


        if switch_frame is not None:
            if count < switch_frame or switch_frame == -1:
                label = 0  # red traffic light
            else:
                label = 1  # green traffic light
            image_path = pj(images_dir, '{:03}_{}_{}.jpg'.format(count, video_name, label))
        else:
            image_path = pj(images_dir, '{:03}_{}.jpg'.format(count, video_name))
        cv2.imwrite(image_path, image)

        success, image = video_capture.read()
        count += 1


def parse_image_name(image_name):
    image_name = os.path.splitext(image_name)[0]  # delete file's extension
    image_name_splitted = image_name.split('_')

    frame, video_name, label = int(image_name_splitted[0]), image_name_splitted[1], int(image_name_splitted[2])

    return frame, video_name, label


def create_classification_dir_from_images_dirs(images_dirs, classification_dir):
    images_dirs = filter(os.path.isdir, images_dirs)

    images_paths = []
    for images_dir in images_dirs:
        images_paths.extend(glob(pj(images_dir ,'*.jpg')))

    if not os.path.exists(classification_dir):
        os.mkdir(classification_dir)
        os.mkdir(pj(classification_dir, '0'))
        os.mkdir(pj(classification_dir, '1'))

        for image_path in tqdm(images_paths) :
            frame, video_name, label = parse_image_name(os.path.basename(image_path))
            shutil.copy(image_path, pj(classification_dir, str(label)))
    else:
        print('Directory {} already exists!'.format(classification_dir))


def load_switch_frames(path):
    if os.path.exists(path):
        with open(path) as fin:
            video_name_to_switch_frame = dict()
            for line in fin.readlines():
                line_splitted = line.strip().split(' ')
                video_name, switch_frame = line_splitted[0], int(line_splitted[-1])

                video_name_to_switch_frame[video_name] = switch_frame
        return video_name_to_switch_frame
    else:
        return None


def create_images_from_videos(videos_dir, images_from_videos_dir, video_name_to_switch_frame=None):
    if not os.path.exists(images_from_videos_dir):
        os.mkdir(images_from_videos_dir)

        video_paths = list(filter(lambda x: x.endswith('.avi'), list_dir_with_full_paths(videos_dir)))
        for video_path in tqdm(video_paths):
            video_base_name = os.path.basename(video_path)
            images_dir = pj(images_from_videos_dir, os.path.splitext(video_base_name)[0])
            os.mkdir(images_dir)

            if video_name_to_switch_frame is not None:
                switch_frame = video_name_to_switch_frame[video_base_name]
                extract_images_from_video(video_path, images_dir, switch_frame)
            else:
                extract_images_from_video(video_path, images_dir)
    else:
        print('Directory {} already exists!'.format(images_from_videos_dir))


def multiple_copy(srcs, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)

    for src in srcs:
        shutil.copytree(src, pj(dst, os.path.basename(src)))


# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--videos-dir', type=str,
                    help='path to dir where video data is stored')
parser.add_argument('--prepared-data-dir', type=str,
                    help='path to dir where prepared data will be stored')
parser.add_argument('--val-ratio', default=0.2, type=float,
                    help='ratio of validation dataset. 0 if not splitting is not needed')

args = parser.parse_args()

if __name__ == '__main__':
    video_name_to_switch_frame = load_switch_frames(pj(args.videos_dir, 'ideal.txt'))

    images_from_videos_dir = pj(args.prepared_data_dir, 'images_from_videos')

    print('Creating images from videos...')
    create_images_from_videos(args.videos_dir, images_from_videos_dir, video_name_to_switch_frame)

    if args.val_ratio != 0:
        train_images_dirs, val_images_dirs = train_test_split(
            list_dir_with_full_paths(images_from_videos_dir),
            test_size=args.val_ratio, random_state=0
        )

        multiple_copy(train_images_dirs, pj(args.prepared_data_dir, 'train_images_from_videos'))
        multiple_copy(val_images_dirs, pj(args.prepared_data_dir, 'val_images_from_videos'))

        print('Creating train classification dir...')
        create_classification_dir_from_images_dirs(train_images_dirs,
                                                   pj(args.prepared_data_dir, 'train_images_by_class'))

        print('Creating train classification dir...')
        create_classification_dir_from_images_dirs(val_images_dirs,
                                                   pj(args.prepared_data_dir, 'val_images_by_class'))
