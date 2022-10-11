'''
Merges depth conversion, segmentation correction, and video creation in one script,
and performs it in parallel across multiple scenes.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields (CVPR 2022).
'''

import argparse
import colorsys
import cv2
import imageio
import json
import matplotlib
import matplotlib.pyplot as plt
import minexr
import multiprocessing
import numpy as np
import os
import platform
import shutil
import skvideo.io
import sys
import time
import tqdm

_DEFAULT_ROOT_DIR = r'greater_4d/'
_DEPTH_SRC_SUFFIX = '_depth.exr'
_SEGM_SRC_SUFFIX = '_preflat.png'
_SNITCH_SRC_SUFFIX = '_preflat_snitch.png'
_VIEW_NAMES = ['View 1', 'View 2', 'View 3', 'View 4', 'View 5']
_NEW_DEPTH_EXR = True

parser = argparse.ArgumentParser()


def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument(
    '--root_dir', default=_DEFAULT_ROOT_DIR, type=str,
    help='Generated GREATER dataset base directory.')
parser.add_argument(
    '--num_frames', default=181, type=int,
    help='Number of frames per video. (default: 181)')
parser.add_argument(
    '--num_views', default=3, type=int,
    help='Number of expected views per scene. (default: 3)')
parser.add_argument(
    '--speed_factor', default=1, type=int,
    help='Dataset speed factor. (default: 1)')
parser.add_argument(
    '--fps', default=24, type=int,
    help='Frame rate of scene videos to export. (default: 24)')
parser.add_argument(
    '--ignore_if_exist', default=True, type=str2bool,
    help='If True, do not process files if their destinations already exist.')
parser.add_argument(
    '--remove_depth_exr', default=True, type=str2bool,
    help='If True, remove raw source depth maps after conversion to PNG to save space.')
parser.add_argument(
    '--max_depth_clip', default=32.0, type=float,
    help='Upper range of depth map. Ensure that (almost) no depth pixel is larger than this value in meters. (default: 32.0)')
parser.add_argument(
    '--num_processes', default=4, type=int,
    help='Number of parallel workers on the scene level. (default: 4)')
parser.add_argument(
    '--write_poses', default=False, type=str2bool,
    help='If True, draw 3x4 camera pose matrix values (extrinsics) onto every frame of the video.')
parser.add_argument(
    '--mark_snitch_occl_cont', default=True, type=str2bool,
    help='If True, process detailed snitch occlusion & containment info per frame, per view.')
parser.add_argument(
    '--index_from', default=-1, type=int,
    help='If >= 0, lower bound of range of scenes to process (inclusive).')
parser.add_argument(
    '--index_to', default=-1, type=int,
    help='If >= 0, upper bound of range of scenes to process (inclusive).')

# Directory structure:
# root -> train / val / test.
#      -> GREATER_000000 / GREATER_000001 / etc.
#      -> images_view1 / images_view2 / etc.
#      -> 1234.png, 1234_depth.png, 1234_preflat.png, etc.


def draw_text(image, x, y, label, size=0.65, color_scale_value=None):
    # Draw background and write text using OpenCV.
    label_width = int((25 + len(label) * 15) * size)
    label_height = 24
    image[y:y+label_height, x:x+label_width] = (0, 0, 0)
    if color_scale_value is None:
        color = (255, 255, 255)  # Plain white.
    else:
        # Interpolate between (0, 255, 0) and (255, 63, 63)
        color = (int(color_scale_value * 255.0),
                 int(63 + (1.0 - color_scale_value) * 192.0),
                 int(color_scale_value * 63.0))
    image = cv2.putText(image, label, (x, y+label_height-8), 2, size, color, 1)
    return image


def convert_depth_old(depth_src_fp, args):
    depth_src = np.load(depth_src_fp)
    # depth = depth[:, :, 0]  # OLD: First 3 channels are equal, 4th is empty.
    depth = depth_src[::-1, :]  # Vertically flipped for some reason.
    depth /= args.max_depth_clip
    depth = np.clip(depth, 0.0, 1.0)
    # Turn into black to make it clear that out-of-range values are invalid.
    depth[depth == 1.0] = 0.0
    depth *= 65535.0  # == 2^16 - 1
    depth = depth.astype(np.uint16)
    return depth


def convert_depth_new(depth_src_fp, args):
    with open(depth_src_fp, 'rb') as filep:
        reader = minexr.load(filep)
    depth = reader.image  # (H, 1, W).
    depth = depth.squeeze().copy()
    depth /= args.max_depth_clip
    depth = np.clip(depth, 0.0, 1.0)
    # Turn into black to make it clear that out-of-range values are invalid.
    depth[depth == 1.0] = 0.0
    depth *= 65535.0  # == 2^16 - 1
    depth = depth.astype(np.uint16)
    return depth


def augment_segm_snitch_xray(segm_src, snitch_src):
    '''
    Assumes snitch has a red component.
    '''
    result = segm_src.copy()
    mask = np.zeros_like(snitch_src[:, :, 0], dtype=np.bool)
    mask[1:-1, :] = (snitch_src[2:, :, 0] != snitch_src[:-2, :, 0])
    mask[1:-1, :] = (snitch_src[:-2, :, 0] != snitch_src[2:, :, 0])
    mask[:, 1:-1] = (snitch_src[:, 2:, 0] != snitch_src[:, :-2, 0])
    mask[:, 1:-1] = (snitch_src[:, :-2, 0] != snitch_src[:, 2:, 0])
    result[mask] = [1, 1, 1]
    return result


def calculate_occlusion(all_obj_segm, snitch_segm):
    '''
    Determines the fraction of the snitch that is occluded from a particular view, per frame.
    Args:
        all_obj_segm: (H, W, 3) numpy array of flat scene image (all objects visible).
        snitch_segm: (H, W, 3) numpy array of flat scene image (only snitch visible).
    Returns:
        (num_frames) numpy float array, with values in [0, 1].
    '''
    snitch_mask = snitch_segm.sum(axis=2) >= 0.05
    snitch_mask_count = snitch_mask.sum()
    visible = np.abs(all_obj_segm[snitch_mask] - snitch_segm[snitch_mask]).sum(axis=-1) <= 0.05
    visible_count = visible.sum()
    visible_fraction = visible_count / (snitch_mask_count + 1e-9) * (1.0 + 1e-9)
    return 1.0 - visible_fraction


def retrieve_containments(scene_json_fp, proximities, num_frames, speed_factor):
    '''
    Determines for every frame whether the snitch is contained by another object.
    Args:
        scene_json_fp: Full file path to Blender scene information stored as JSON.
        num_frames: Number of frames
    Returns:
        (num_frames) numpy float array, with values in [0, 1].
    '''
    with open(scene_json_fp, 'r') as f:
        scene_struct = json.load(f)
    result = np.zeros(num_frames // speed_factor + 1)

    # Look at actions to determine *future* containment.
    movements = scene_struct['movements']
    for obj_id in movements.keys():
        cur_actions = movements[obj_id]
        # Example: [['_contain', 'Cone_0', 9, 39], ['_slide', None, 40, 60]].
        for action in cur_actions:
            # Example: ['_contain', 'Cone_0', 9, 39].
            action_name = action[0].lower()
            if 'contain' in action_name:
                contained_obj = action[1].lower()
                if 'spl' in contained_obj:
                    print('Snitch contained!')
                    print(scene_json_fp)
                    print(action)
                    frame_start = min(int(action[2] / speed_factor), len(result) - 1)
                    frame_end = min(int(action[3] / speed_factor), len(result) - 1)
                    for i in range(frame_start, frame_end + 1):
                        # Snitch is about to be contained by one more object at this time.
                        result[i] += 0.5
    
    # Look at nearest distance to determine *present* containment.
    result[:len(proximities)] += (proximities < 0.60) * 1.0
    
    return result


def retrieve_proximities(prox_fp, num_frames, speed_factor):
    result = np.loadtxt(prox_fp)
    assert num_frames // speed_factor <= len(result) <= num_frames // speed_factor + 1
    return result


def draw_pose_matrix(image, x, y, matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            image = draw_text(image, x + j * 72, y + i * 32, f'{matrix[i, j]:.1f}', size=0.6)
    return image


def postprocess_loop(args, scn_idx_start, scn_idx_step):
    cur_scene_idx = -1

    split_dns = sorted(os.listdir(args.root_dir))
    split_dns.sort()
    if any(['_' in split_dn for split_dn in split_dns]):
        split_dns = ['']  # Probably no splits are used.
    for split_dn in split_dns:  # train / val / test.
        split_dp = os.path.join(args.root_dir, split_dn)
        if not os.path.isdir(split_dp):
            continue
        # print(split_dp)

        scene_dns = sorted(os.listdir(split_dp))
        scene_dns.sort()
        for scene_dn in scene_dns:  # GREATER_000000 / GREATER_000001 / etc.
            scene_dp = os.path.join(split_dp, scene_dn)
            if not os.path.isdir(scene_dp):
                continue
            # Ensure we are restricted to relevant scenes for this process.
            cur_scene_idx += 1
            if (cur_scene_idx - scn_idx_start) % scn_idx_step != 0:
                continue
            # Ensure the dataset index is within range.
            dset_idx = int(scene_dn[-6:])
            if args.index_from >= 0 and dset_idx < args.index_from:
                continue
            if args.index_to >= 0 and dset_idx > args.index_to:
                continue
            print(scene_dp)

            images_dirs = [dn for dn in os.listdir(scene_dp) if 'images' in dn
                           and not 'depth_tmp' in dn]
            images_dirs.sort()
            num_views = len(images_dirs)
            if num_views < args.num_views:
                print('This scene has not yet been fully generated.')

            final_frames = []
            dst_fp = os.path.join(scene_dp, f'video{cur_scene_idx:05d}.mp4')

            if os.path.exists(dst_fp) and os.path.isfile(dst_fp) and args.ignore_if_exist:
                print(dst_fp, 'already exists!')
                continue

            all_occlusions = []

            if args.mark_snitch_occl_cont:
                scene_json_fp = os.path.join(scene_dp, 'scene_view0.json')
                prox_fp = os.path.join(scene_dp, 'poses_view1', 'snitch_proximities.txt')
                proximities = retrieve_proximities(prox_fp, args.num_frames, args.speed_factor)
                containments = retrieve_containments(
                    scene_json_fp, proximities, args.num_frames, args.speed_factor)

            for view_idx, image_dn in enumerate(images_dirs):  # images_view1 / images_view2 / etc.
                image_dp = os.path.join(scene_dp, image_dn)
                if not os.path.isdir(image_dp):
                    continue
                print(image_dp)

                # Load list of camera extrinsics for this view.
                if num_views == 1:
                    pose_fp = os.path.join(scene_dp, 'poses', 'camera_RT.npy')
                else:
                    pose_fp = os.path.join(scene_dp, f'poses_view{view_idx + 1}', 'camera_RT.npy')
                camera_RT = np.load(pose_fp)

                occlusions = []

                # Iterate over all frames.
                all_files = os.listdir(image_dp)
                all_files.sort()
                if len(all_files) < 10:
                    continue
                frame_idx = 0

                if args.num_processes == 1:
                    to_iterate = tqdm.tqdm(all_files)
                else:
                    to_iterate = all_files

                for image_fn in to_iterate:
                    # Expected format: 1234.png
                    if len(image_fn) > 8 or image_fn[-4:].lower() != '.png':
                        continue

                    src_fn = image_fn[:-4]
                    src_fp = os.path.join(image_dp, src_fn)
                    try:
                        rgb_src_fp = src_fp + '.png'
                        rgb_src = plt.imread(rgb_src_fp)[:, :, :3]
                        depth_src_fp = src_fp + _DEPTH_SRC_SUFFIX
                        depth_dst_fp = src_fp + '_depth.png'
                        segm_src_fp = src_fp + _SEGM_SRC_SUFFIX
                        snitch_src_fp = src_fp + _SNITCH_SRC_SUFFIX

                    except Exception as e:
                        print('Failed for:', src_fp)
                        print(e)
                        time.sleep(0.5)
                        continue

                    # Convert and save depth map if needed.
                    if os.path.exists(depth_src_fp) and os.path.isfile(depth_src_fp):
                        if _NEW_DEPTH_EXR:
                            depth_dst = convert_depth_new(depth_src_fp, args)
                        else:
                            depth_dst = convert_depth_old(depth_src_fp, args)
                        # depth_dst is of type uint16.
                        imageio.imwrite(depth_dst_fp, depth_dst)
                        if args.remove_depth_exr:
                            os.remove(depth_src_fp)

                    # Read depth from file.
                    # NOTE: Got error here, not a PNG file?
                    depth_src = plt.imread(depth_dst_fp)

                    # Read segmentation from file.
                    segm_src = plt.imread(segm_src_fp)[:, :, :3]

                    # Read snitch-only segmentation from file.
                    snitch_src = plt.imread(snitch_src_fp)[:, :, :3]

                    # Augment segm with snitch for video visualization.
                    segm_src = augment_segm_snitch_xray(segm_src, snitch_src)

                    if args.mark_snitch_occl_cont:
                        cur_occl_frac = calculate_occlusion(segm_src, snitch_src)
                        cur_cont = containments[frame_idx]
                        occlusions.append(cur_occl_frac)

                    # Construct visual frame.
                    width, height = rgb_src.shape[1], rgb_src.shape[0]
                    # print(width, height, rgb_src.dtype)
                    left_offset = (292 if args.write_poses else 0)
                    if view_idx == 0:
                        video_width = width * 3 + left_offset
                        video_height = height * num_views
                        frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
                    else:
                        frame = final_frames[frame_idx]
                    y1 = height * view_idx
                    y2 = height * (view_idx + 1)

                    frame[y1:y2, left_offset:left_offset+width] = \
                        np.uint8(rgb_src[:, :, :3] * 255.0)
                    frame[y1:y2, left_offset+width:left_offset+width*2] = \
                        np.uint8(np.tile(depth_src[:, :, np.newaxis], (1, 1, 3)) * 255.0)
                    frame[y1:y2, -width:] = np.uint8(segm_src[:, :, :3] * 255.0)

                    if left_offset > 0:
                        frame = draw_text(frame, 4, y1 + 4, _VIEW_NAMES[view_idx])
                    else:
                        frame = draw_text(frame, 0, y1, _VIEW_NAMES[view_idx])

                    if args.write_poses:
                        frame = draw_pose_matrix(frame, 4, y1 + 44, camera_RT[frame_idx])

                    if args.mark_snitch_occl_cont:
                        frame = draw_text(frame, 4, y1 + height - 90,
                                          f'Sn Occl: {int(cur_occl_frac * 100.0):d}%',
                                          color_scale_value=cur_occl_frac)
                        frame = draw_text(frame, 4, y1 + height - 62,
                                          f'Sn Cont: ' +
                                          ('Yes' if cur_cont >= 1.0 else
                                           'Soon' if cur_cont >= 0.5 else 'No'),
                                          color_scale_value=np.clip(cur_cont, 0.0, 1.0))
                        frame = draw_text(frame, 4, y1 + height - 34,
                                          f'Sn Prox: {proximities[frame_idx]:.2f}',
                                          color_scale_value=np.clip(1.0 - proximities[frame_idx] / 4.0, 0.0, 1.0))

                    if view_idx == 0:
                        final_frames.append(frame)
                    frame_idx += 1

                all_occlusions.append(occlusions)

            # writer = skvideo.io.FFmpegWriter(dst_fp, inputdict={'-r': str(args.fps)})
            # for frame in final_frames:
            #     writer.writeFrame(frame)
            # writer.close()
            imageio.mimwrite(dst_fp, final_frames, fps=args.fps, quality=10)

            occl_dst_fp = os.path.join(scene_dp, 'occl.txt')
            cont_dst_fp = os.path.join(scene_dp, 'cont.txt')
            np.savetxt(occl_dst_fp, all_occlusions)
            np.savetxt(cont_dst_fp, containments)
            
            print('Done for:', dst_fp)


def main(args):
    processes = []
    for process_idx in range(args.num_processes):
        process = multiprocessing.Process(
            target=postprocess_loop, args=(args, process_idx, args.num_processes))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    print('Done!')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
