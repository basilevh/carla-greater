'''
Visualizes a captured scene by generating a video with RGB frames over different views.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields (CVPR 2022).
'''

# Library imports.
import argparse
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import sys
import tqdm

# Internal imports.
import my_utils
from my_utils import str2bool


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--capture_path', type=str,
                        help='Directory path containing raw sensor data.')
    parser.add_argument('--video_path', type=str,
                        help='Output file path to write the MP4 video to.')
    parser.add_argument('--ignore_if_exist', default=True, type=str2bool,
                        help='Halt if the output files already exist.')
    parser.add_argument('--num_frames', default=1010, type=int,
                        help='Number of frames to process, starting from zero.')
    parser.add_argument('--fps', default=10, type=int,
                        help='Frame rate of the exported video.')
    parser.add_argument('--only_forward_clean', default=False, type=str2bool,
                        help='Ignore other views, and omit labels.')

    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    if args.video_path is None:
        file_name = str(pathlib.Path(args.capture_path).name) + '_video'
        file_name += '_multiview'
        file_name += '.mp4'
        args.video_path = os.path.join(args.capture_path, file_name)
        print('Set video_path:', args.video_path)

    if args.ignore_if_exist and os.path.exists(args.video_path) and \
            os.path.isfile(args.video_path) and \
            os.path.getsize(args.video_path) >= 2 * 2048 * 2048:
        print(f'Video file path {args.video_path} already exists and is at least 2 MB!')
        print('Exiting...')
        sys.exit(0)

    if args.only_forward_clean:
        data_prefix = 'mv_raw_'
        image_types = ['forward_rgb']
        gallery_height, gallery_width = 1, 1
        reference_idx = -1

    else:
        data_prefix = 'mv_raw_'
        image_types = ['magic_left_rgb', 'magic_right_rgb',
                       'forward_rgb', 'magic_top_rgb']
        gallery_height, gallery_width = 2, 2
        reference_idx = -1

    gallery_count = gallery_height * gallery_width

    frames = []
    for frame_idx in tqdm.tqdm(range(args.num_frames)):
        gallery_images = []

        for image_type in image_types:

            if image_type is None:
                # Insert black (empty) tile to organize visualization.
                image = np.zeros((height, width, 3), dtype=np.uint8)
                gallery_images.append(image)
                continue

            src_fp = os.path.join(
                args.capture_path, data_prefix + 'all', f'{frame_idx:05d}_{image_type}.png')

            if os.path.exists(src_fp):
                image = plt.imread(src_fp)
                (height, width) = image.shape[:2]
                if len(image.shape) == 3:
                    # RGB or semantic segmentation.
                    image = image[..., :3]  # Ignore alpha.
                    image = image[..., ::-1]  # BGR -> RGB.
                else:
                    # Depth => single channel (no extra dimension).
                    image = 1.0 / (image * 16.0 + 1.0)  # Convert to disparity.
                    image = np.tile(image[:, :, None], (1, 1, 3))
                image = (image * 255.0).astype(np.uint8)
                gallery_images.append(image)

        if len(gallery_images) == 0:
            print('Nothing found? Last src_fp:')
            print(src_fp)
            continue

        if reference_idx >= 0:
            ref_image = gallery_images[reference_idx].astype(np.int16)  # Contains no objects.
        else:
            ref_image = None

        total_height = gallery_height * height
        total_width = gallery_width * width
        frame = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        for gallery_idx in range(gallery_count):
            image_type = image_types[gallery_idx % len(image_types)]
            
            if image_type is not None:
                label = 'all_' + '_'.join(image_type.split('_')[:-1])
            else:
                label = ''
            
            grid_x = gallery_idx % gallery_width
            grid_y = gallery_idx // gallery_width
            
            if reference_idx < 0 or gallery_idx == 0 or gallery_idx == reference_idx:
                show = gallery_images[gallery_idx]
            else:
                show = np.abs((gallery_images[gallery_idx] - ref_image) * 4).astype(np.uint8)
            
            if not(args.only_forward_clean):
                show = my_utils.draw_text(show, 0, 0, label, (255, 255, 255), 1.0)
            
            frame[grid_y * height:(grid_y + 1) * height,
                  grid_x * width:(grid_x + 1) * width] = show

        frames.append(frame)

    if len(frames) != 0:
        output_parent = str(pathlib.Path(args.video_path).parent)
        os.makedirs(output_parent, exist_ok=True)
        my_utils.write_video(args.video_path, frames, args.fps)


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print()
        print('Done!')
