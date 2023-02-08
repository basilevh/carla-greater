'''
Originally created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
Feel free to adapt this code as it shows how to correctly use the CARLA-4D dataset in PyTorch pipelines.
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import torch
import torch_cluster
import torchvision

_MAX_DEPTH_CLIP = 1000.0


class CARLADataset(torch.utils.data.Dataset):
    '''
    Multiview CARLA dataset for object permanence.
    Assumes directory & file structure:
    dataset_root\train\train_01234\mv_raw_all\01234_forward_rgb.png + 01234_forward_lidar.npy + ...
    For clarity, this class is a simplified implementation compared to the paper.
    '''

    @staticmethod
    def max_depth_clip():
        return _MAX_DEPTH_CLIP

    def __init__(self, dataset_root, logger, stage='train', video_length=4, frame_skip=4,
                 n_points_rnd=8192, n_fps_input=1024, n_fps_target=1024, pcl_input_frames=3,
                 pcl_target_frames=1, reference_frame=None, correct_origin_ground=True, min_z=-1.0,
                 other_bounds=20.0, target_bounds=16.0):
        '''
        :param dataset_root (str): Path to dataset or single scene.
        :param stage (str): Subfolder (dataset split) to use; may remain empty.
        :param n_points_rnd (int): Number of points to retain after initial random subsampling.
            This is applied almost directly after converting the RGB-D data to a point cloud.
        :param n_fps_input, n_fps_target (int): If > 0, number of points to retain after farthest
            point sampling of the input and target point clouds respectively after processing for
            time and/or view aggregation.
        :param pcl_input_frames (int): Number of input frames to show, counting from the beginning.
        :param pcl_target_frames (int): Number of target frames to provide, counting from the end.
            Typically, video_length <= pcl_input_frames + pcl_target_frames.
        :param reference_frame (int): If not None, time index to use as common coordinate system for
            all point clouds. This is typically last input frame (i.e. present).
        :param correct_origin_ground (bool): Translate all lidar measurements in the Z direction
            such that the origin equals the ground, instead of existing relative to the car height.
        :param other_bounds (= pt_cube_bounds) (float): Input cube bounds.
        :param target_bounds (= cr_cube_bouds) (float): Output & target cube bounds.
        '''
        self.dataset_root = dataset_root
        self.logger = logger
        self.stage = stage
        self.video_length = video_length
        self.frame_skip = frame_skip
        self.n_points_rnd = n_points_rnd
        self.n_fps_input = n_fps_input
        self.n_fps_target = n_fps_target
        self.pcl_input_frames = pcl_input_frames
        self.pcl_target_frames = pcl_target_frames
        self.reference_frame = reference_frame
        self.correct_origin_ground = correct_origin_ground
        self.min_z = min_z
        self.other_bounds = other_bounds
        self.target_bounds = target_bounds

        self.stage_dir = os.path.join(dataset_root, stage)
        if not os.path.exists(self.stage_dir):
            self.stage_dir = dataset_root  # We may already be pointing to the stage directory.
            self.dataset_root = str(pathlib.Path(dataset_root).parent)

        all_scenes = os.listdir(self.stage_dir)
        all_scenes = [dn for dn in all_scenes if '_' in dn
                        and os.path.isdir(os.path.join(self.stage_dir, dn))]
        all_scenes.sort()

        self.all_scenes = all_scenes
        self.num_scenes = len(all_scenes)
        self.dset_size = self.num_scenes

        self.min_input_size = 64
        self.min_target_size = 512

        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return self.dset_size

    def _get_frame_start(self, scene_dp):
        '''
        :return (frame_start, num_frames).
        '''
        scene_content_dp = os.path.join(scene_dp, 'mv_raw_all')
        image_fns = [fn for fn in os.listdir(scene_content_dp) if 'forward_rgb' in fn]
        num_frames = len(image_fns)

        frame_start_low = 10
        frame_end_high = num_frames - 20
        frame_start_high = frame_end_high - self.video_length * self.frame_skip
        frame_start = np.random.randint(frame_start_low, frame_start_high)

        return (frame_start, num_frames)

    def __getitem__(self, index):
        '''
        :return Dictionary with all information for a single example.
        '''
        scene_idx = index % self.num_scenes  # Avoid same index for subsequent examples.
        scene_dn = self.all_scenes[scene_idx]
        scene_dp = os.path.join(self.stage_dir, scene_dn)

        video_fp = os.path.join(scene_dp, scene_dn + '_video_multiview.mp4')
        if not os.path.exists(video_fp):
            raise RuntimeError('Video file does not exist: {}'.format(video_fp))
        scene_content_dp = os.path.join(scene_dp, 'mv_raw_all')
        sensor_matrices_fp = os.path.join(scene_content_dp, 'sensor_matrices.npy')
        if not os.path.exists(sensor_matrices_fp):
            raise RuntimeError('Sensor matrices file does not exist: {}'.format(sensor_matrices_fp))

        sensor_RT = np.load(os.path.join(scene_content_dp, 'sensor_matrices.npy'))
        sensor_RT = sensor_RT.astype(np.float32)  # (T, V, 4, 4) = (2010, 8, 4, 4).
        sensor_K = np.load(os.path.join(scene_content_dp, 'camera_K.npy'))
        sensor_K = sensor_K.astype(np.float32)  # (3, 3).

        # List of sensors != list of views, so use hard-coded correspondence.
        view_sensor_matching = [0, 3, 4, 5]
        view_names = ['forward', 'magic_left', 'magic_right', 'magic_top']
        num_views = len(view_names)
        sensor_RT = sensor_RT[:, view_sensor_matching]  # (T, V, 4, 4) = (2010, 4, 4, 4).

        # Make a random selection of temporal clip bounds within available video.
        (frame_start, num_frames) = self._get_frame_start(scene_dp)
        frame_end = frame_start + self.video_length * self.frame_skip
        frame_inds = np.arange(frame_start, frame_end, self.frame_skip)

        all_rgb = []
        all_RT = []
        all_K = []
        all_lidar = []

        for v in range(num_views):
            view = view_names[v]
            view_rgb = []
            view_RT = []
            view_K = []
            view_lidar = []

            for f in frame_inds:
                rgb_fp = os.path.join(scene_content_dp, f'{f:05d}_{view}_rgb.png')
                lidar_segm_fp = os.path.join(
                    scene_content_dp, f'{f:05d}_{view}_lidar_segm.npy')

                rgb = plt.imread(rgb_fp)[..., :3].astype(np.float32)
                cam_RT = sensor_RT[f, v].astype(np.float32)  # (4, 4).
                cam_K = sensor_K.astype(np.float32)  # (3, 3).
                lidar = np.load(lidar_segm_fp).astype(np.float32)  # (N, 9).
                # (x, y, z, cosine_angle, instance_id, semantic_tag, R, G, B).

                # Transform to common reference frame (= present time, forward view).
                if self.reference_frame is not None:
                    ref_frame_idx = frame_inds[self.reference_frame]
                else:
                    ref_frame_idx = f
                ref_view_idx = 0

                if f != ref_frame_idx or v != ref_view_idx:
                    source_matrix = cam_RT
                    target_matrix = sensor_RT[ref_frame_idx, ref_view_idx]
                    target_matrix = target_matrix.astype(np.float32)
                    lidar = transform_lidar_frame(lidar, source_matrix, target_matrix)

                # Translate in Z to ensure the sensor origin equals the ground, such that
                # min_z indicates distance to street level. However, the street surface
                # level can also vary across the map, so we cannot rely on sensor_RT, and
                # instead we have to use the hard-coded sensor height of 1 meter during
                # dataset construction.
                if self.correct_origin_ground:
                    ref_sensor_height = 1.0
                    lidar[..., 2] += ref_sensor_height

                # Restrict to cuboid of interest.
                lidar = filter_pcl_bounds_carla_input_numpy(
                    lidar, min_z=self.min_z, other_bounds=self.other_bounds,
                    cube_mode=self.cube_mode)

                # NOTE: This step has no effect if there are insufficient points.
                if self.n_points_rnd > 0:
                    lidar = subsample_pad_pcl_numpy(
                        lidar, self.n_points_rnd, subsample_only=False)
                lidar = lidar.astype(np.float32)
                # NOTE: At this stage, we have only used primitive subsampling techniques,
                # and the point cloud is still generally oversized. Later, we use FPS and
                # match sizes taking into account how frames and/or views are aggregated.

                view_rgb.append(rgb)
                view_RT.append(cam_RT)
                view_K.append(cam_K)
                view_lidar.append(lidar)

            view_rgb = np.stack(view_rgb)  # (T, H, W, 3).
            view_RT = np.stack(view_RT)  # (T, 3, 4).
            view_K = np.stack(view_K)  # (T, 3, 3).
            # NOTE: We cannot stack view_lidar because of potentially different sizes.
            # view_lidar = List-T of (N, 9).

            all_rgb.append(view_rgb)
            all_RT.append(view_RT)
            all_K.append(view_K)
            all_lidar.append(view_lidar)

        all_rgb = np.stack(all_rgb)  # (V, T, H, W, 3).
        all_RT = np.stack(all_RT)  # (V, T, 3, 4).
        all_K = np.stack(all_K)  # (V, T, 3, 3).
        # all_lidar = List-V of List-T of (N, 9).

        # Generate appropriate versions of the point cloud data.
        (V, T) = (num_views, self.video_length)
        all_pcl_sizes = np.array([[all_lidar[v][t].shape[0]
                                    for t in range(T)] for v in range(V)])
        # List-V of List-T of (N, 9) with
        # (x, y, z, cosine_angle, instance_id, semantic_tag, R, G, B).
        lidar_video_views = accumulate_pcl_time_numpy(all_lidar)
        # List-V of (T*N, 10) with
        # (x, y, z, cosine_angle, instance_id, semantic_tag, R, G, B, t).
        lidar_merged_frames = merge_pcl_views_numpy(all_lidar, insert_view_idx=True)
        # List-T of (V*N, 10) with
        # (x, y, z, cosine_angle, instance_id, semantic_tag, view_idx, R, G, B).

        # Limit input to the desired time range.
        if self.pcl_input_frames < self.video_length:
            show_frame_size_sum = 0
            for t in range(self.pcl_input_frames):
                show_frame_size_sum += all_lidar[0][t].shape[0]
            pcl_input = lidar_video_views[0][:show_frame_size_sum]
        else:
            pcl_input = lidar_video_views[0]
        # (x, y, z, cosine_angle, instance_id, semantic_tag, R, G, B, t).

        # Always shuffle point cloud data just before converting to tensor.
        np.random.shuffle(pcl_input)
        pcl_input = self.to_tensor(pcl_input).squeeze(0)  # (T*N, 10).

        # Subsample random input video and merged target here for efficiency.
        pre_sample_size = pcl_input.shape[0]
        pcl_input = subsample_pad_pcl_torch(
            pcl_input, self.n_fps_input, sample_mode='farthest_point', subsample_only=False)
        post_sample_size = pcl_input.shape[0]
        pcl_input_size = min(pre_sample_size, post_sample_size)

        # NOTE: The input may sometimes empty if we restrict point categories.
        # We have to skip these cases because they are unreliable.
        if pcl_input_size < self.min_input_size:
            raise RuntimeError(f'Invalid due to pcl_input_size: {pcl_input_size}')

        pcl_target = []  # List-T of (V*N, 10).
        pcl_target_size = []
        for t in range(self.pcl_target_frames):
            pcl_target_frame = lidar_merged_frames[-self.pcl_target_frames + t]
            np.random.shuffle(pcl_target_frame)
            pcl_target_frame = self.to_tensor(pcl_target_frame).squeeze(0)  # (V*N, 10).
            # (x, y, z, cosine_angle, instance_id, semantic_tag, view_idx, R, G, B).

            # It is best to filter the target by the output cube here already.
            # Retain 2 meters of context to allow for semantic guidance during supervision.
            padding = 2.0
            pcl_target_frame = filter_pcl_bounds_carla_output_torch(
                pcl_target_frame, min_z=self.min_z, other_bounds=self.target_bounds,
                padding=padding, cube_mode=self.cube_mode)

            pcl_target.append(pcl_target_frame)
            pcl_target_size.append(pcl_target_frame.shape[0])

        # NOTE: The target is sometimes empty in CARLA.
        # We have to skip these cases because they are unreliable.
        if np.any(np.array(pcl_target_size) < self.min_target_size):
            raise RuntimeError(f'Invalid due to pcl_target_size: {pcl_target_size}')

        if self.n_fps_target != 0:
            # NOTE: farthest_point is relatively expensive, while random is fast but less spatially
            # balanced.
            sample_mode = 'farthest_point' if self.n_fps_target > 0 else 'random'

            for i in range(self.pcl_target_frames):
                pre_sample_size = pcl_target[i].shape[0]
                pcl_target[i] = subsample_pad_pcl_torch(
                    pcl_target[i], abs(self.n_fps_target),
                    sample_mode=sample_mode, subsample_only=False)
                post_sample_size = pcl_target[i].shape[0]
                pcl_target_size[i] = min(pre_sample_size, post_sample_size)

        else:
            # Do not further subsample target point cloud.
            assert pcl_target[0].shape[0] == pcl_target_size[0]

        # Ensure cosine angle, instance id, and semantic tag are kept separate in input view.
        pcl_input_sem = pcl_input[..., 3:-4]
        # (N, 3) with (cosine_angle, instance_id, semantic_tag).
        pcl_input = torch.cat([pcl_input[..., :3], pcl_input[..., -4:]], dim=-1)
        # (N, 7) with (x, y, z, R, G, B, t).

        # Metadata is all lightweight stuff (so no big arrays or tensors).
        meta_data = dict()
        meta_data['data_kind'] = 1002  # Cannot be string.
        meta_data['num_views'] = num_views
        meta_data['num_frames'] = num_frames  # Total in this scene video.
        meta_data['scene_idx'] = scene_idx
        # Clip subselection, e.g. [464, 467, 470, 473, 476, 479].
        meta_data['frame_inds'] = frame_inds
        meta_data['n_fps_input'] = self.n_fps_input
        meta_data['n_fps_target'] = self.n_fps_target
        meta_data['pcl_sizes'] = all_pcl_sizes  # Per view and per frame.
        meta_data['pcl_input_size'] = pcl_input_size
        meta_data['pcl_target_size'] = pcl_target_size
        meta_data['view_sensor_matching'] = view_sensor_matching

        # Make all information easily accessible.
        to_return = dict()
        to_return['rgb'] = all_rgb
        to_return['cam_RT'] = all_RT
        to_return['cam_K'] = all_K
        to_return['pcl_input'] = pcl_input
        # (N, 8) with (x, y, z, R, G, B, t, mark_track).
        to_return['pcl_input_sem'] = pcl_input_sem
        # (N, 3) with (cosine_angle, instance_id, semantic_tag).
        to_return['pcl_target'] = pcl_target
        # List of (M, 11) with (x, y, z, cosine_angle, instance_id, semantic_tag, view_idx, R, G, B, mark_track).
        to_return['meta_data'] = meta_data

        return to_return


def merge_intensity_semantic_lidar(lidar, lidar_segm):
    '''
    :param lidar (N, 7) numpy array with (x, y, z, intensity, R, G, B).
    :param lidar_segm (N, 9) numpy array with
        (x, y, z, cosine_angle, instance_id, semantic_tag, R, G, B).
    :return lidar_merge (N, 10) numpy array with
        (x, y, z, intensity, cosine_angle, instance_id, semantic_tag, R, G, B).
    '''
    assert lidar.shape[0] == lidar_segm.shape[0]
    np.testing.assert_array_almost_equal(lidar[0, :3], lidar_segm[0, :3])
    np.testing.assert_array_almost_equal(lidar[-1, :3], lidar_segm[-1, :3])
    result_xyzi = lidar[..., :4]
    result_sem = lidar_segm[..., 3:-3]
    result_rgb = lidar[..., -3:]
    result = np.concatenate([result_xyzi, result_sem, result_rgb], axis=-1)
    return result


def transform_lidar_frame(lidar_pcl, source_matrix, target_matrix):
    '''
    Converts the coordinates of the measured point cloud data from one coordinate frame to another.
    :param lidar_pcl (N, D) numpy array with rows (x, y, z, *).
    :param source_matrix (4, 4) numpy array.
    :param target_matrix (4, 4) numpy array.
    :return transformed_pcl (N, D) numpy array with rows (x, y, z, *).
    '''
    (N, D) = lidar_pcl.shape
    inv_target_matrix = np.linalg.inv(target_matrix)

    pcl_xyz = lidar_pcl[..., :3].T  # (3, N).
    points_source = np.concatenate([pcl_xyz, np.ones_like(pcl_xyz[:1])], axis=0)  # (4, N).
    points_world = np.dot(source_matrix, points_source)  # (4, N).
    points_target = np.dot(inv_target_matrix, points_world)  # (4, N).

    pcl_xyz = points_target[:3].T  # (N, 3).
    transformed_pcl = lidar_pcl.copy()
    transformed_pcl[..., :3] = pcl_xyz

    return transformed_pcl


def filter_pcl_bounds_numpy(pcl, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0,
                            z_min=-10.0, z_max=10.0):
    '''
    Restricts a point cloud to exclude coordinates outside a certain cube.
    :param pcl (N, D) numpy array: Point cloud with first 3 elements per row = (x, y, z).
    :return (N, D) numpy array: Filtered point cloud.
    '''
    mask_x = np.logical_and(x_min <= pcl[..., 0], pcl[..., 0] <= x_max)
    mask_y = np.logical_and(y_min <= pcl[..., 1], pcl[..., 1] <= y_max)
    mask_z = np.logical_and(z_min <= pcl[..., 2], pcl[..., 2] <= z_max)
    mask_xy = np.logical_and(mask_x, mask_y)
    mask_xyz = np.logical_and(mask_xy, mask_z)
    mask = mask_xyz

    result = pcl[mask]
    return result


def filter_pcl_bounds_carla_input_numpy(pcl, min_z=-0.5, other_bounds=20.0):
    '''
    Restricts a point cloud to exclude coordinates outside a certain cube.
    This method is tailored to the CARLA dataset.
    :param pcl (N, D) numpy array: Point cloud with first 3 elements per row = (x, y, z).
    :param min_z (float): Discard everything spatially below this value.
    :param other_bounds (float): Input data cube bounds for the point transformer.
    :return (N, D) numpy array: Filtered point cloud.
    '''
    pcl = filter_pcl_bounds_numpy(
        pcl, x_min=-other_bounds * 0.7, x_max=other_bounds * 2.5, y_min=-other_bounds * 1.0,
        y_max=other_bounds * 1.0, z_min=min_z, z_max=other_bounds * 0.5)

    return pcl


def subsample_pad_pcl_numpy(pcl, n_desired, subsample_only=False):
    '''
    If the point cloud is too small, leave as is (nothing changes).
    If the point cloud is too large, subsample uniformly randomly.
    :param pcl (N, D) numpy array.
    :param n_desired (int).
    :param sample_mode (str): random or farthest_point.
    :param subsample_only (bool): If True, do not allow padding.
    :return (n_desired, D) numpy array.
    '''
    N = pcl.shape[0]

    if N < n_desired:
        if subsample_only:
            raise RuntimeError('Too few input points: ' +
                               str(N) + ' vs ' + str(n_desired) + '.')

        return pcl
        # zeros = np.zeros((n_desired - N, pcl.shape[1]), dtype=pcl.dtype)
        # result = np.concatenate((pcl, zeros), axis=0)
        # return result

    elif N > n_desired:
        inds = np.random.choice(N, n_desired, replace=False)
        inds.sort()
        result = pcl[inds]
        return result

    else:
        return pcl


def subsample_pad_pcl_torch(pcl, n_desired, sample_mode='random', subsample_only=False):
    '''
    If the point cloud is too small, apply zero padding.
    If the point cloud is too large, subsample either uniformly randomly or by farthest point
        sampling (FPS) per batch item.
    :param pcl (B, N, D) tensor.
    :param n_desired (int).
    :param sample_mode (str): random or farthest_point.
    :param subsample_only (bool): If True, do not allow padding.
    :param retain_vehped (bool): Do not subsample cars & people (semantic tags 4 and 10).
    :param segm_idx (int): Semantic tag index.
    :return (B, n_desired, D) tensor.
    '''
    assert sample_mode in ['random', 'farthest_point']
    no_batch = (len(pcl.shape) == 2)
    if no_batch:
        pcl = pcl.unsqueeze(0)
    (B, N, D) = pcl.shape

    if N < n_desired:
        if subsample_only:
            raise RuntimeError('Too few input points: ' +
                               str(N) + ' vs ' + str(n_desired) + '.')

        zeros = torch.zeros((B, n_desired - N, D), dtype=pcl.dtype)
        zeros = zeros.to(pcl.device)
        result = torch.cat((pcl, zeros), axis=1)
        if no_batch:
            result = result.squeeze(0)
        return result

    elif N > n_desired:
        n_remain = n_desired

        assert B == 1
        remain_inds = np.arange(N)  # (N).

        result = torch.zeros((B, n_remain, D), dtype=pcl.dtype)

        if sample_mode == 'random':
            for i in range(B):
                inds = np.random.choice(remain_inds, n_remain, replace=False)
                inds.sort()
                result[i] = pcl[i, inds]

        else:  # farthest_point.
            pcl_flat = pcl.view(B * N, D)
            coords_flat = pcl_flat[..., :3]
            batch = torch.arange(B).repeat_interleave(N)  # (B*N).
            batch = batch.to(pcl.device)
            # NOTE: This fps call has inherent randomness!
            inds = torch_cluster.fps(
                src=coords_flat, batch=batch, ratio=n_remain / N - 1e-7)
            inds = torch.sort(inds)[0]
            pcl_sub_flat = pcl_flat[inds]
            result = pcl_sub_flat.view(B, n_remain, D)

        if no_batch:
            result = result.squeeze(0)

        assert result.shape[0] == n_desired
        return result

    else:
        if no_batch:
            pcl = pcl.squeeze(0)
        return pcl


def accumulate_pcl_time_numpy(pcl):
    '''
    Converts a series of RGB point cloud snapshots into a point cloud video by adding a feature that
        represents time with values {0, 1, ..., T-1}.
    :param pcl (V, T, N, D) numpy array or list-V of list-T of (N, D) numpy arrays.
    :return (V, T*N, D+1) numpy array or list-V of (T*N, D+1) numpy arrays.
    '''
    if isinstance(pcl, np.ndarray):
        # Fully within the numpy array domain.
        (V, T, N, D) = pcl.shape
        time_vals = np.arange(T, dtype=np.float32)[None, :, None, None]  # (1, T, 1, 1).
        time_vals = np.tile(time_vals, (V, 1, N, 1))  # (V, T, N, 1).
        pcl_out = np.concatenate((pcl, time_vals), axis=-1)  # (V, T, N, 7).
        pcl_out = pcl_out.reshape(V, T * N, D + 1)

    else:
        # Mixed list and array domain, which is more complicated but more flexible.
        (V, T) = len(pcl), len(pcl[0])
        pcl_out = []
        for view_idx in range(V):
            pcl_view = []
            for time_idx in range(T):
                pcl_frame = pcl[view_idx][time_idx]  # (N, 6).
                time_vals = np.ones_like(pcl_frame[..., 0:1]) * time_idx  # (N, 1).
                pcl_frame_timed = np.concatenate([pcl_frame, time_vals], axis=-1)  # (N, 7).
                pcl_view.append(pcl_frame_timed)
            pcl_view = np.concatenate(pcl_view, axis=0)  # (T*N, 7).
            pcl_out.append(pcl_view)

    return pcl_out


def merge_pcl_views_numpy(pcl, insert_view_idx=False):
    '''
    Converts a set of RGB point clouds from different camera viewpoints into one combined point
        cloud.
    :param pcl (V, T, N, D) numpy array or list-V of list-T of (N, D) numpy arrays.
    :return (T, V*N, D) numpy array or list-T of (V*N, D) numpy arrays.
    '''
    if isinstance(pcl, np.ndarray):
        # Fully within the numpy array domain.
        assert not insert_view_idx
        (V, T, N, D) = pcl.shape
        pcl_out = pcl.transpose(1, 0, 2, 3)  # (T, V, N, 6).
        pcl_out = pcl_out.reshape(T, V * N, D)

    else:
        # Mixed list and array domain, which is more complicated but more flexible.
        V, T = len(pcl), len(pcl[0])
        pcl_out = []
        for time_idx in range(T):
            pcl_time = []

            for view_idx in range(V):
                cur_xyz_sem = pcl[view_idx][time_idx][..., :-3]
                cur_rgb = pcl[view_idx][time_idx][..., -3:]

                if insert_view_idx:
                    cur_idx = np.ones_like(cur_xyz_sem[..., 0:1]) * view_idx
                    pcl_time_view = np.concatenate([cur_xyz_sem, cur_idx, cur_rgb], axis=-1)

                else:
                    pcl_time_view = np.concatenate([cur_xyz_sem, cur_rgb], axis=-1)

                pcl_time.append(pcl_time_view)

            pcl_time = np.concatenate(pcl_time, axis=0)  # (V*N, 6).
            pcl_out.append(pcl_time)

    return pcl_out


def filter_pcl_bounds_torch(pcl, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0,
                            z_min=-10.0, z_max=10.0):
    '''
    Restricts a point cloud to exclude coordinates outside a certain cube.
    :param pcl (B, N, D) tensor: Point cloud with first 3 elements per row = (x, y, z).
    :return (B, N, D) tensor: Filtered point cloud.
    '''
    mask_x = torch.logical_and(x_min <= pcl[..., 0], pcl[..., 0] <= x_max)
    mask_y = torch.logical_and(y_min <= pcl[..., 1], pcl[..., 1] <= y_max)
    mask_z = torch.logical_and(z_min <= pcl[..., 2], pcl[..., 2] <= z_max)
    mask_xy = torch.logical_and(mask_x, mask_y)
    mask_xyz = torch.logical_and(mask_xy, mask_z)
    result = pcl[mask_xyz]
    return result


def filter_pcl_bounds_carla_output_torch(pcl, min_z=-0.5, other_bounds=16.0, padding=0.0):
    '''
    Restricts a point cloud to exclude coordinates outside a certain cube.
    This method is tailored to the CARLA dataset.
    :param pcl (B, N, D) tensor: Point cloud with first 3 elements per row = (x, y, z).
    :param min_z (float): Discard everything spatially below this value.
    :param other_bounds (float): Output data cube bounds for the point transformer.
    :param padding (float): Still include this buffer in 5 directions for context.
    :return (B, N, D) tensor: Filtered point cloud.
    '''
    pcl = filter_pcl_bounds_torch(
        pcl, x_min=0.0 - padding, x_max=other_bounds * 2.5 + padding,
        y_min=-other_bounds * 1.0 - padding, y_max=other_bounds * 1.0 + padding,
        z_min=min_z, z_max=other_bounds * 0.4)

    return pcl
