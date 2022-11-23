'''
Analysis of point clouds to eventually select interesting clips within CARLA scenes with
more occlusion etc.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields (CVPR 2022).
'''


# Library imports.
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tqdm

# Internal imports.
import my_utils
from my_utils import str2bool


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dset_root', default='path/to/carla_4d/', type=str)
    parser.add_argument('--split', default='val', type=str)
    parser.add_argument('--scene_idx', default=-1, type=int)
    parser.add_argument('--min_z', default=-2.0, type=float)
    parser.add_argument('--cr_cube_bounds', default=16.0, type=float)
    parser.add_argument('--cube_mode', default=4, type=int)
    parser.add_argument('--ignore_if_exist', default=False, type=str2bool)
    parser.add_argument('--horizontal_fov', default=120.0, type=float)
    parser.add_argument('--frame_step', default=3, type=int)

    args = parser.parse_args()
    return args


def count_near_forward_border(lidar, horizontal_fov, min_x):
    '''
    Returns the number of points that are (almost) out of frame as seen from the forward lidar.
    '''
    atan = np.arctan2(lidar[..., 1], lidar[..., 0])
    atan = np.abs(atan * 180.0 / np.pi)
    # near_border_fov = (np.abs(atan - horizontal_fov / 2.0) <= 4.0)
    near_border_fov = (atan >= horizontal_fov / 2.0 - 6.0)
    near_border_cube = (lidar[..., 0] <= min_x + 0.2)
    near_border = np.logical_or(near_border_fov, near_border_cube)
    count = near_border.sum()
    return count


def count_near_rear_border(lidar, max_x):
    '''
    Returns the number of points that are either out of bounds, or about disappear in the distance.
    '''
    near_border = lidar[..., 0] >= max_x - 0.8
    count = near_border.sum()
    return count


def count_near_side_border(lidar, max_y):
    '''
    Returns the number of points that are either out of bounds, or about to disappear left or right.
    '''
    near_border = np.logical_or(lidar[..., 1] >= max_y - 0.6,
                                lidar[..., 1] <= -max_y + 0.6)
    count = near_border.sum()
    return count


def process_scene(args, scene_dp, scene_dn):

    dst_fp = os.path.join(scene_dp, f'occlusion_rate_fs{args.frame_step}_cm{args.cube_mode}.npy')
    if os.path.exists(dst_fp) and args.ignore_if_exist:
        print(dst_fp, 'exists! Skipping...')
        return False

    correct_origin_ground = True
    min_z = args.min_z
    cube_mode = args.cube_mode

    # view_names = ['forward', 'magic_left', 'magic_right', 'magic_top']
    # view_sensor_matching = [0, 3, 4, 5]
    # view_names = ['forward', 'magic_top']
    # view_sensor_matching = [0, 5]
    view_names = ['forward']
    view_sensor_matching = [0]

    video_fp = os.path.join(scene_dp, scene_dn + '_video_multiview.mp4')
    scene_content_dp = os.path.join(scene_dp, 'mv_raw_all')
    sensor_matrices_fp = os.path.join(scene_content_dp, 'sensor_matrices.npy')

    image_fns = [fn for fn in os.listdir(scene_content_dp)
                 if 'forward_rgb' in fn and fn[0] != '.' and fn[-4:] == '.png']
    num_views = len(view_names)
    # num_sensors = len(sensor_types)
    num_frames = len(image_fns)
    frame_low = 0
    frame_high = num_frames
    frame_skip = 1
    # frame_skip = 10

    sensor_RT = np.load(os.path.join(scene_content_dp, 'sensor_matrices.npy'))
    sensor_RT = sensor_RT.astype(np.float32)  # (T, V, 4, 4) = (2010, 8, 4, 4).
    sensor_K = np.load(os.path.join(scene_content_dp, 'camera_K.npy'))
    sensor_K = sensor_K.astype(np.float32)  # (3, 3).
    sensor_names = list(np.genfromtxt(
        os.path.join(scene_content_dp, 'sensor_names.txt'), dtype='str'))

    sensor_RT = sensor_RT[:, view_sensor_matching]  # (T, V, 4, 4) = (2010, 4, 4, 4).

    all_RT = []
    all_K = []
    all_lidar = []
    frame_inds = np.arange(frame_low, frame_high, frame_skip)
    print('frame_inds:')
    print(frame_inds[:4], '...', frame_inds[-4:])

    for v in range(num_views):
        view = view_names[v]
        view_RT = []
        view_K = []
        view_lidar = []
        print('view:', view)

        for f in tqdm.tqdm(frame_inds):
            lidar_fp = os.path.join(
                scene_content_dp, f'{f:05d}_{view}_lidar_segm.npy')

            cam_RT = sensor_RT[f, v].astype(np.float32)  # (4, 4).
            cam_K = sensor_K.astype(np.float32)  # (3, 3).
            lidar = np.load(lidar_fp).astype(np.float32)  # (N, 9).
            # (x, y, z, cosine_angle, instance_id, semantic_tag, R, G, B).

            # Transform to common reference frame.
            # ref_frame_idx = 0  # NOTE: This is the very beginning of the entire video!
            ref_frame_idx = f  # Works better for cube filtering.
            ref_view_idx = 0
            if f != ref_frame_idx or v != ref_view_idx:
                source_matrix = cam_RT
                target_matrix = sensor_RT[ref_frame_idx, ref_view_idx]
                target_matrix = target_matrix.astype(np.float32)
                lidar = my_utils.transform_lidar_frame(
                    lidar, source_matrix, target_matrix)

            # Translate in Z to ensure the sensor origin equals the ground, such that
            # min_z indicates distance to street level.
            if correct_origin_ground:
                ref_sensor_height = 1.0
                if ref_sensor_height <= 0.0 or ref_sensor_height >= 2.0:
                    raise RuntimeError(
                        'Unexpected ref_sensor_height:', ref_sensor_height)
                lidar[..., 2] += ref_sensor_height

            # Restrict to cuboid of interest.
            lidar = my_utils.filter_pcl_bounds_carla_output_numpy(
                lidar, min_z=min_z, other_bounds=args.cr_cube_bounds, cube_mode=cube_mode)

            lidar = lidar.astype(np.float32)

            view_RT.append(cam_RT)
            view_K.append(cam_K)
            view_lidar.append(lidar)

        view_RT = np.stack(view_RT)  # (T, 3, 4).
        view_K = np.stack(view_K)  # (T, 3, 3).
        # NOTE: We cannot stack view_lidar because of potentially different sizes.

        all_RT.append(view_RT)
        all_K.append(view_K)
        all_lidar.append(view_lidar)

    all_RT = np.stack(all_RT)  # (V, T, 3, 4).
    all_K = np.stack(all_K)  # (V, T, 3, 3).
    # all_lidar = List-V of List-T of (N, 9).
    # (x, y, z, cosine_angle, instance_id, semantic_tag, R, G, B).
    # https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
    # 4 = pedestrian, 10 = vehicle, 20 = dynamic.
    interesting_tags = [4, 10, 20]
    K = len(interesting_tags)

    (V, T) = (num_views, len(frame_inds))
    all_lidar_sizes = np.array([[all_lidar[v][t].shape[0]
                                for t in range(T)] for v in range(V)])
    print('(V, T):', (V, T))
    print('all_lidar_sizes:')
    print(all_lidar_sizes.mean(axis=1))

    # Count occurence (i.e. number of points) of all instances (object IDs) of interest over time.
    instance_points_dict = dict()  # Maps semantic_tag to category_points_dict.
    for tag in interesting_tags:

        # Maps object ID to (V, T, 2) array, i.e. point counts (both total and near borders)
        # per frame per view.
        category_points_dict = dict()

        for t, f in enumerate(frame_inds):
            for v in range(num_views):
                lidar = all_lidar[v][t]
                lidar_category = lidar[lidar[..., 5] == tag]

                instance_ids = np.unique(lidar_category[..., 4]).astype(np.uint32)
                for id in instance_ids:
                    lidar_instance = lidar_category[lidar_category[..., 4] == id]

                    if id not in category_points_dict:
                        category_points_dict[id] = np.zeros((V, T, 3), dtype=np.uint32)

                    # Save total number of points, nearly out of frame points, and next out of frame
                    # points.
                    assert args.cube_mode == 4
                    min_x = 0.0
                    max_x = args.cr_cube_bounds * 2.5
                    max_y = args.cr_cube_bounds * 1.0
                    category_points_dict[id][v, t, 0] = lidar_instance.shape[0]
                    category_points_dict[id][v, t, 1] = \
                        count_near_forward_border(lidar_instance, args.horizontal_fov, min_x) + \
                        count_near_rear_border(lidar_instance, max_x) + \
                        count_near_side_border(lidar_instance, max_y)

                    # Safeguard against moving or turning ego cars by also counting the out of frame
                    # points when the current point cloud is transformed to the next reference
                    # frame. For example, when any object is about to become invisible, this step
                    # will treat it as already out of frame.
                    if t < T - 1:
                        source_matrix = sensor_RT[f, 0].astype(np.float32)
                        target_matrix = sensor_RT[f + 1, 0].astype(np.float32)
                        lidar_shift = my_utils.transform_lidar_frame(
                            lidar_instance, source_matrix, target_matrix)
                        category_points_dict[id][v, t, 2] = \
                            count_near_forward_border(lidar_shift, args.horizontal_fov, min_x) + \
                            count_near_rear_border(lidar_shift, max_x) + \
                            count_near_side_border(lidar_shift, max_y)

        instance_points_dict[tag] = category_points_dict

    # Now, we can access e.g. instance_points_dict[4][922][0, 42] to get the number of lidar points
    # for category 4 instance 922 at view 0 frame 42.
    # Next, define the occlusion rate as the proportion of an object that disappears from the
    # past to the current frame (relative to the highest point count of an instance ever seen
    # from that view), which is a value between 0 and 1. Disocclusions are ignored.
    # The total occlusion rate sums the occlusion rate over all objects of a particular category in
    # the scene. We try to filter for in-frame cases by looking at point coordinates.
    occlusion_rate = np.zeros((K, V, T, 3))  # Last 3 = (raw, clipped, inframe).

    for k, tag in enumerate(interesting_tags):

        for id in instance_points_dict[tag].keys():

            # Mark frames with this instance near any border.
            bad_frames = np.zeros(T, dtype=np.int32)
            bad_frames += (instance_points_dict[tag][id][0, :, 1] > 0)
            for offset in range(1, args.frame_step + 1):
                bad_frames[offset:] += (instance_points_dict[tag][id][0, :-offset, 1] > 0)
                bad_frames[offset:] += (instance_points_dict[tag][id][0, :-offset, 2] > 0)
            bad_frames = (bad_frames > 0)

            for v in range(num_views):
                max_inst_points = instance_points_dict[tag][id][v, :, 0].max()
                rel_inst_points = instance_points_dict[tag][id][v, :, 0] / \
                    (max_inst_points + 1e-6)
                rel_inst_points_delta = np.zeros(T, dtype=np.float32)
                rel_inst_points_delta[args.frame_step:] = \
                    rel_inst_points[args.frame_step:] - rel_inst_points[:-args.frame_step]

                cur_raw_occlusion_rate = -rel_inst_points_delta
                cur_clip_occlusion_rate = np.clip(cur_raw_occlusion_rate, 0.0, 1.0)
                cur_inframe_occlusion_rate = cur_clip_occlusion_rate.copy()

                # Filter out bad frames (only for this specific instance id).
                cur_inframe_occlusion_rate[bad_frames] = 0.0

                occlusion_rate[k, v, :, 0] += cur_raw_occlusion_rate
                occlusion_rate[k, v, :, 1] += cur_clip_occlusion_rate
                occlusion_rate[k, v, :, 2] += cur_inframe_occlusion_rate

    # Example: access occlusion_rate[0, 0, 35, 2] for how many pedestrians are disappearing
    # from the forward view between frame 34 and 35, excluding those going out-of-frame.
    # First index = K = choice between pedestrian / vehicle / dynamic.
    # Last index = 3 = choice between raw / clipped / inframe.

    np.save(dst_fp, occlusion_rate)

    print('Stored occlusion_rate to:', dst_fp)
    print()

    return True


def main():

    args = get_args()

    split_dp = os.path.join(args.dset_root, args.split)

    all_scenes = os.listdir(split_dp)
    all_scenes = [dn for dn in all_scenes if '_' in dn
                  and os.path.isdir(os.path.join(split_dp, dn))]
    all_scenes.sort()
    num_scenes = len(all_scenes)

    if args.scene_idx >= 0:
        scene_inds = [args.scene_idx]
    else:
        scene_inds = np.arange(num_scenes)

    for scene_idx in scene_inds:

        try:
            print('scene_idx:', scene_idx)
            scene_dn = all_scenes[scene_idx]
            scene_dp = os.path.join(split_dp, scene_dn)
            print('scene_dn:', scene_dn)
            result = process_scene(args, scene_dp, scene_dn)

        except Exception as e:
            print(e)


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)

    main()
