'''
Originally created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
Feel free to adapt this code as it shows how to correctly use the GREATER dataset in PyTorch pipelines.
'''

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import torch
import torch_cluster
import torchvision

_MAX_DEPTH_CLIP = 32.0

# Values are ints in the range [0, 360] degrees.
_PREFLAT_HUE_CLUSTERS = [0, 35, 47, 65, 90, 160, 180, 188, 219, 284, 302, 324]


def get_occlusion_rate(scene_dp, src_view):
    snitch_occl_fp = os.path.join(scene_dp, 'occl.txt')
    snitch_occl = np.loadtxt(snitch_occl_fp)  # (T).
    snitch_occl = snitch_occl[src_view]

    frame_step = 3
    occlusion_rate = np.zeros_like(snitch_occl)
    occlusion_rate[frame_step:] = snitch_occl[frame_step:] - snitch_occl[:-frame_step]
    occlusion_rate = np.clip(occlusion_rate, 0.0, 1.0)

    return occlusion_rate


class GREATERDataset(torch.utils.data.Dataset):
    '''
    Assumes directory & file structure:
    dataset_root\train\GREATER_000012\images_view2\0123.png + 0123_depth.png + 0123_preflat.png.
    For clarity, this class is a simplified implementation compared to the paper.
    '''

    @staticmethod
    def max_depth_clip():
        return _MAX_DEPTH_CLIP

    @staticmethod
    def preflat_hue_clusters():
        return _PREFLAT_HUE_CLUSTERS

    def __init__(self, dataset_root, logger, stage='train', video_length=4, frame_skip=4,
                 convert_to_pcl=True, n_points_rnd=8192, n_fps_input=1024, n_fps_target=1024,
                 pcl_input_frames=3, pcl_target_frames=1, min_z=-1.0, other_bounds=5.0,
                 return_segm=True):
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
        :param min_z (float): Ignore points outside this cube.
        :param other_bounds (float) = max_z, min_x, max_x, min_y, max_y.
        :param return_segm (bool): Whether to load instance IDs in addition to RGB.
        '''
        self.dataset_root = dataset_root
        self.logger = logger
        self.stage = stage
        self.video_length = video_length
        self.frame_skip = frame_skip
        self.convert_to_pcl = convert_to_pcl
        self.n_points_rnd = n_points_rnd
        self.n_fps_input = n_fps_input
        self.n_fps_target = n_fps_target
        self.pcl_input_frames = pcl_input_frames
        self.pcl_target_frames = pcl_target_frames
        self.min_z = min_z
        self.other_bounds = other_bounds
        self.return_segm = return_segm

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
        self.dset_size = int(self.num_scenes)

        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return self.dset_size

    def _get_frame_start(self, scene_dp):
        '''
        :return (frame_start, num_frames).
        '''
        image_dp = os.path.join(scene_dp, 'images_view1')
        rgb_fns = [fn for fn in os.listdir(image_dp) if fn[-4:] == '.png' and len(fn) <= 8]
        num_frames = len(rgb_fns)
        frame_start = np.random.randint(0, num_frames - self.video_length * self.frame_skip)

        return (frame_start, num_frames)

    def __getitem__(self, index):
        '''
        :return Dictionary with all information for a single example.
        '''
        scene_idx = index
        scene_dp = os.path.join(self.stage_dir, self.all_scenes[scene_idx])
        image_dns = [dn for dn in os.listdir(scene_dp) if 'images' in dn]
        image_dns.sort()
        image_dps = [os.path.join(scene_dp, dn) for dn in image_dns]
        pose_dns = [dn for dn in os.listdir(scene_dp) if 'poses' in dn]
        pose_dns.sort()
        pose_dps = [os.path.join(scene_dp, dn) for dn in pose_dns]
        assert len(image_dns) == len(pose_dns)

        # Make a random selection of input view index for future use.
        num_views = len(image_dns)
        src_view = np.random.randint(num_views)

        # Make a random selection of temporal clip bounds within available video.
        (frame_start, num_frames) = self._get_frame_start(scene_dp)
        frame_end = frame_start + self.video_length * self.frame_skip
        frame_inds = np.arange(frame_start, frame_end, self.frame_skip)

        all_rgb = []
        all_depth = []
        all_flat = []
        all_snitch = []
        all_RT = []
        all_K = []
        all_pcl = []

        for v in range(num_views):
            view_rgb = []
            view_depth = []
            view_flat = []
            view_snitch = []
            view_RT = []
            view_K = []
            view_pcl = []

            src_RT_fp = os.path.join(pose_dps[v], 'camera_RT.npy')
            src_K_fp = os.path.join(pose_dps[v], 'camera_K.npy')
            src_RT = np.load(src_RT_fp)
            src_K = np.load(src_K_fp)

            for f in frame_inds:
                rgb_fp = os.path.join(image_dps[v], f'{f:04d}.png')
                flat_fp = os.path.join(image_dps[v], f'{f:04d}_preflat.png')
                depth_fp = os.path.join(image_dps[v], f'{f:04d}_depth.png')

                rgb = plt.imread(rgb_fp)[..., :3].astype(np.float32)
                flat = plt.imread(flat_fp)[..., :3].astype(np.float32)
                depth = plt.imread(depth_fp).astype(np.float32) * _MAX_DEPTH_CLIP
                cam_RT = src_RT[f].astype(np.float32)
                cam_K = src_K[f].astype(np.float32)

                # NOTE: Important fix here! (due to the way Blender uses camera matrix data)
                cam_K[1, 1] = cam_K[0, 0]

                view_rgb.append(rgb)
                view_depth.append(depth)
                view_flat.append(flat)
                view_RT.append(cam_RT)
                view_K.append(cam_K)

                if self.return_segm:
                    snitch_fp = os.path.join(image_dps[v], f'{f:04d}_preflat_snitch.png')
                    snitch = plt.imread(snitch_fp)
                    view_snitch.append(snitch)

            view_rgb = np.stack(view_rgb)  # (T, H, W, 3).
            view_depth = np.stack(view_depth)  # (T, H, W).
            view_flat = np.stack(view_flat)  # (T, H, W, 3).
            view_snitch = np.stack(view_snitch) if self.return_segm else []  # (T, H, W, 3).
            view_RT = np.stack(view_RT)  # (T, 3, 4).
            view_K = np.stack(view_K)  # (T, 3, 3).

            # Extract point clouds.
            for f in range(len(frame_inds)):
                rgb = view_rgb[f]
                flat = view_flat[f]
                depth = view_depth[f]
                cam_RT = view_RT[f]
                cam_K = view_K[f]

                # Extract instance_id from flat by rounding to the nearest known hue cluster.
                flat_hsv = matplotlib.colors.rgb_to_hsv(flat)
                inst_ids = np.round(flat_hsv[..., 0:1] * 360.0)  # (H, W, 1).
                inst_ids = np.abs(inst_ids[..., None] - _PREFLAT_HUE_CLUSTERS)  # (H, W, 12).
                inst_ids = inst_ids.argmin(axis=-1)  # (H, W, 1).
                inst_ids[flat_hsv[..., 1] < 0.9] = -1.0  # Background or floor is irrelevant.

                # Incorporate both color and instance segmentation info in the point cloud.
                rgb_inst = np.concatenate([inst_ids, rgb], axis=-1)  # (H, W, 4).
                pcl_full = point_cloud_from_rgbd(rgb_inst, depth, cam_RT, cam_K)
                pcl_full = pcl_full.astype(np.float32)
                # (N, 7) with (x, y, z, instance_id, R, G, B).

                # Restrict to cuboid of interest.
                # NOTE: min_z >= 0.1 means discard the entire floor.
                pcl_full = filter_pcl_bounds_numpy(
                    pcl_full, x_min=-self.other_bounds, x_max=self.other_bounds,
                    y_min=-self.other_bounds, y_max=self.other_bounds,
                    z_min=self.min_z, z_max=self.other_bounds, greater_floor_fix=True)

                # NOTE: This step has no effect if there are insufficient points.
                if self.n_points_rnd > 0:
                    pcl_full = subsample_pad_pcl_numpy(
                        pcl_full, self.n_points_rnd, subsample_only=False)
                # NOTE: At this stage, we have only used primitive subsampling techniques, and
                # the point cloud is still generally oversized. Later, we use FPS and match
                # sizes taking into account how frames and/or views are aggregated.

                view_pcl.append(pcl_full)

            all_rgb.append(view_rgb)
            all_depth.append(view_depth)
            all_flat.append(view_flat)
            all_snitch.append(view_snitch)
            all_RT.append(view_RT)
            all_K.append(view_K)
            all_pcl.append(view_pcl)

        all_rgb = np.stack(all_rgb)  # (V, T, H, W, 3).
        all_depth = np.stack(all_depth)  # (V, T, H, W).
        all_flat = np.stack(all_flat)  # (V, T, H, W, 3).
        all_snitch = np.stack(all_snitch) if self.return_segm else []  # (V, T, H, W, 3).
        all_RT = np.stack(all_RT)  # (V, T, 3, 4).
        all_K = np.stack(all_K)  # (V, T, 3, 3).

        # Generate appropriate versions of the point cloud data.
        (V, T) = (num_views, self.video_length)
        all_pcl_sizes = np.array([[all_pcl[v][t].shape[0] for t in range(T)] for v in range(V)])
        # List-V of List-T of (N, 7) with (x, y, z, instance_id, R, G, B).
        pcl_video_views = accumulate_pcl_time_numpy(all_pcl)
        # List-V of (T*N, 8) with (x, y, z, instance_id, R, G, B, t).
        pcl_merged_frames = merge_pcl_views_numpy(all_pcl, insert_view_idx=True)
        # List-T of (V*N, 8) with (x, y, z, instance_id, view_idx, R, G, B).

        # Limit input to the desired time range.
        if self.pcl_input_frames < self.video_length:
            show_frame_size_sum = 0
            for t in range(self.pcl_input_frames):
                show_frame_size_sum += all_pcl[src_view][t].shape[0]
            pcl_input = pcl_video_views[src_view][:show_frame_size_sum]
        else:
            pcl_input = pcl_video_views[src_view]
        # (x, y, z, instance_id, R, G, B, t).

        # Always shuffle point cloud data just before converting to tensor.
        np.random.shuffle(pcl_input)
        pcl_input = self.to_tensor(pcl_input).squeeze(0)  # (T*N, 8).

        # Subsample random input video and merged target here for efficiency.
        pre_sample_size = pcl_input.shape[0]
        pcl_input = subsample_pad_pcl_torch(
            pcl_input, self.n_fps_input, sample_mode='farthest_point', subsample_only=False)
        post_sample_size = pcl_input.shape[0]
        pcl_input_size = min(pre_sample_size, post_sample_size)

        pcl_target = []  # List-T of (V*N, 7).
        pcl_target_size = []
        for t in range(self.pcl_target_frames):
            pcl_target_frame = pcl_merged_frames[-self.pcl_target_frames + t]
            np.random.shuffle(pcl_target_frame)
            pcl_target_frame = self.to_tensor(pcl_target_frame).squeeze(0)  # (V*N, 8).
            # (x, y, z, instance_id, view_idx, R, G, B).
            pcl_target.append(pcl_target_frame)
            pcl_target_size.append(pcl_target_frame.shape[0])

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
            for i in range(self.pcl_target_frames):
                assert pcl_target[i].shape[0] == pcl_target_size[i]

        # Ensure instance id is kept separate in input view.
        pcl_input_sem = pcl_input[..., 3:-4]
        # (N, 1) with (instance_id).
        pcl_input = torch.cat([pcl_input[..., :3], pcl_input[..., -4:]], dim=-1)
        # (N, 7) with (x, y, z, R, G, B, t).

        # Metadata is all lightweight stuff (so no big arrays or tensors).
        meta_data = dict()
        meta_data['num_views'] = num_views
        meta_data['num_frames'] = num_frames  # Total in this scene video.
        meta_data['scene_idx'] = scene_idx
        meta_data['frame_inds'] = frame_inds  # Clip subselection, e.g. [88, 92, 96, 100].
        meta_data['src_view'] = src_view
        meta_data['n_fps_input'] = self.n_fps_input
        meta_data['n_fps_target'] = self.n_fps_target
        meta_data['pcl_sizes'] = all_pcl_sizes  # Per view and per frame.
        meta_data['pcl_input_size'] = pcl_input_size
        meta_data['pcl_target_size'] = pcl_target_size

        # Make all information easily accessible.
        to_return = dict()
        to_return['rgb'] = all_rgb
        to_return['depth'] = all_depth
        to_return['flat'] = all_flat
        to_return['snitch'] = all_snitch
        to_return['cam_RT'] = all_RT
        to_return['cam_K'] = all_K
        to_return['pcl_input'] = pcl_input  # (N, 7) with (x, y, z, R, G, B, t).
        to_return['pcl_input_sem'] = pcl_input_sem  # (N, 1) with (instance_id).
        to_return['pcl_target'] = pcl_target  # List of (M, 8) with (x, y, z, instance_id, view_idx, R, G, B).
        to_return['meta_data'] = meta_data

        return to_return


def point_cloud_from_rgbd(rgb, depth, cam_RT, cam_K):
    '''
    Converts an image with depth information to a colorized point cloud.

    Args:
        rgb: (H, W, 3) numpy array.
        depth: (H, W) numpy array (represents Z axis offset, not Euclidean distance).
        cam_RT: 3x4 camera extrinsics matrix for source view.
        cam_K: 3x3 camera intrinsics matrix for source view.

    Returns:
        (N, 6) numpy array consisting of 3D world coordinates + RGB values.
    '''
    # First, obtain 3D world coordinates.
    H, W = rgb.shape[:2]
    valid_y, valid_x = np.where(depth > 0.0)  # (N, 3) int each.
    # (H, W) int each.
    all_y, all_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    y = all_y[valid_y, valid_x]  # (N) int.
    x = all_x[valid_y, valid_x]  # (N) int.
    z = depth[valid_y, valid_x]  # (N) float32.
    # (N, 3) float32.
    points = point_cloud_from_pixel_coords(x, y, z, cam_RT, cam_K)

    # Then, attach attributes.
    colors = rgb[valid_y, valid_x]  # (N, 3) float32.
    pcl = np.concatenate((points, colors), axis=1)  # (N, 6) float32.

    return pcl


def point_cloud_from_pixel_coords(x, y, z, cam_RT, cam_K):
    '''
    Converts a set of source pixel coordinates and depth values to 3D world coordinates.
    NOTE: Source coordinates must always be a 1D list or numpy array.

    Args:
        x: List of horizontal integer source positions in [0, width - 1].
        y: List of vertical integer source positions in [0, height - 1].
        z: List of depth values in meters (represents Z axis offset, not Euclidean distance).
        cam_RT: 3x4 camera extrinsics matrix for source view (2D points given).
        cam_K: 3x3 camera intrinsics matrix for source view.

    Returns:
        (N, 3) numpy array consisting of 3D world coordinates.
    '''
    assert len(x) == len(y)
    assert len(x) == len(z)
    N_points = len(x)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    z = np.array(z, dtype=np.float32)

    # Expand all matrices into 4x4 for consistency.
    cam_RT_4x4 = np.eye(4, dtype=np.float32)
    cam_RT_4x4[:3] = cam_RT
    cam_K_4x4 = np.eye(4, dtype=np.float32)
    cam_K_4x4[:3, :3] = cam_K

    # Get 2D pixels in image space: 4 x N.
    coords_src = np.ones((4, N_points), dtype=np.float32)
    coords_src[0, :] = x
    coords_src[1, :] = y

    # Get 3D points in source camera space: (4 x 4) x (4 x N) = (4 x N).
    points_src = np.matmul(np.linalg.inv(cam_K_4x4), coords_src)

    # Scale 3D points by depth value in all dimensions: 4 x N.
    points_src[:3, :] *= z[np.newaxis, :]

    # Transform 3D points to world space: (4 x 4) x (4 x N) = (4 x N).
    points_world = np.matmul(np.linalg.inv(cam_RT_4x4), points_src)

    # Reshape to N x 3.
    points_world = points_world.transpose()[:, :3]

    return points_world


def filter_pcl_bounds_numpy(pcl, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0,
                            z_min=-10.0, z_max=10.0, greater_floor_fix=True):
    '''
    Restricts a point cloud to exclude coordinates outside a certain cube.
    This method is tailored to the GREATER dataset.
    :param pcl (N, D) numpy array: Point cloud with first 3 elements per row = (x, y, z).
    :param greater_floor_fix (bool): If True, remove the weird curving floor in GREATER.
    :return (N, D) numpy array: Filtered point cloud.
    '''
    mask_x = np.logical_and(x_min <= pcl[..., 0], pcl[..., 0] <= x_max)
    mask_y = np.logical_and(y_min <= pcl[..., 1], pcl[..., 1] <= y_max)
    mask_z = np.logical_and(z_min <= pcl[..., 2], pcl[..., 2] <= z_max)
    mask_xy = np.logical_and(mask_x, mask_y)
    mask_xyz = np.logical_and(mask_xy, mask_z)

    if greater_floor_fix:
        inv_pyramid = np.maximum(np.abs(pcl[..., 0]), np.abs(pcl[..., 1]))
        mask_gf = (pcl[..., 2] > (inv_pyramid - 4.5) / 3.5)
        mask = np.logical_and(mask_gf, mask_xyz)
    else:
        mask = mask_xyz

    result = pcl[mask]
    return result


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
            inds = torch_cluster.fps(src=coords_flat, batch=batch, ratio=n_remain / N - 1e-7)
            
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
