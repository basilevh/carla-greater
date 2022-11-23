'''
Various useful methods.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields (CVPR 2022).
'''

# Library imports.
import argparse
try:
    import carla
except:
    pass
import copy
import cv2
import json
import numpy as np
import os
import pathlib
import re
import time


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_args(args, dst_fp):
    dst_dp = str(pathlib.Path(dst_fp).parent)
    os.makedirs(dst_dp, exist_ok=True)
    with open(dst_fp, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)
    bps = list(bps)
    bps.sort(key=lambda x: x.id)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def find_kth_vehicle(world, k=0):
    all_actors = list(world.get_actors())
    all_actors.sort(key=lambda x: x.id)
    cur_idx = 0
    vehicle = None
    
    for actor in all_actors:
        if actor.type_id.startswith('vehicle'):
            
            if cur_idx == k:
                vehicle = actor
                break
            
            # DEBUG:
            # if cur_idx <= 10:
            # if 'audi' in actor.type_id or 'bmw' in actor.type_id or 'mercedes' in actor.type_id:
            #     print(cur_idx, actor.id, actor.type_id, actor.attributes['color'])
            
            cur_idx += 1  # Keep looking.
    
    if vehicle is None:
        print()
        print('find_kth_vehicle: NOT FOUND')
        print(len(all_actors))
        print()
    
    return vehicle


def get_all_vehicles_pedestrians(world):
    all_actors = list(world.get_actors())
    all_actors.sort(key=lambda x: x.id)
    result = []
    for actor in all_actors:
        if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker'):
            result.append(actor)
    return result


def destroy_by_ids(client, ids):
    client.apply_batch([carla.command.DestroyActor(x) for x in ids])


def destroy_spectator(world):
    print('Destroying spectator...')
    all_actors = world.get_actors()
    for actor in all_actors:
        if actor.is_alive and actor.type_id.startswith('spectator'):
            actor.destroy()


def destroy_all_dynamic_actors(client, world):
    print('Destroying all vehicles, walkers, controllers, and sensors...')
    all_actors = world.get_actors()
    batch = []
    for actor in all_actors:
        if actor.is_alive and (actor.type_id.startswith('vehicle') or
                               actor.type_id.startswith('walker') or
                               actor.type_id.startswith('controller') or
                               actor.type_id.startswith('sensor')):
            batch.append(carla.command.DestroyActor(actor.id))
    client.apply_batch(batch)


def get_env_objs(world, visible_enums, toggleable_enums):
    visible_ids = set()
    invisible_ids = set()
    for enum in toggleable_enums:
        env_objs = world.get_environment_objects(enum)
        cur_ids = set([x.id for x in env_objs])
        if enum in visible_enums:
            visible_ids.update(cur_ids)
        else:
            invisible_ids.update(cur_ids)
    return (visible_ids, invisible_ids)


def write_video(file_path, frames, fps):
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()


def read_video(file_path):
    reader = cv2.VideoCapture(file_path)
    frames = []
    while reader.isOpened():
        ret, frame = reader.read()
        if not ret:
            break
        frames.append(frame)
    reader.release()
    return frames


def draw_text(image, x, y, label, color, scale):
    '''
    Draw shaded background and write text using OpenCV.
    '''
    text_scale = 1.0
    label_width = int(14 * text_scale + len(label) * 18 * text_scale)
    label_height = int(44 * text_scale)
    image[y:y + label_height, x:x + label_width] = image[y:y + label_height, x:x + label_width] / 2
    location = (x, y + label_height - int(16 * text_scale))
    image = cv2.putText(image, label, location, 2, text_scale, color, 1)
    return image


def colorize_lidar_pcl(lidar_pcl, rgb_image, lidar_to_world, world_to_camera, camera_K):
    '''
    Aligns the lidar point cloud with pixels from an RGB camera with the same pose.
    Then, colors the visible parts of the point cloud via the found / projected correspondences.
    :param lidar_pcl (N, D) numpy array with rows (x, y, z, *).
    :param rgb_image (H, W, 3) numpy array.
    :param lidar_to_world (4, 4) numpy array.
    :param world_to_camera (4, 4) numpy array.
    :param camera_K (3, 3) numpy array: Camera intrinsics matrix incorporating focal length.
    :return colorized_pcl (N, D + 3) numpy array with rows (*, R, G, B) when within frustum,
        or (*, -1, -1, -1) otherwise.
    '''
    (N, D) = lidar_pcl.shape
    (H, W, _) = rgb_image.shape

    pcl_xyz = lidar_pcl[..., :3].T  # (3, N).

    # Transform from lidar frame to camera frame.
    points_lidar = np.concatenate([pcl_xyz, np.ones_like(pcl_xyz[:1])], axis=0)  # (4, N).
    points_world = np.dot(lidar_to_world, points_lidar)  # (4, N).
    points_camera = np.dot(world_to_camera, points_world)  # (4, N).

    # Change from UE4's coordinate system to a standard camera coordinate system (OpenCV):
    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y
    # This can be achieved by multiplying by the following matrix:
    # [[ 0,  1,  0 ],
    #  [ 0,  0, -1 ],
    #  [ 1,  0,  0 ]]
    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    points_camera_opencv = np.array([
        points_camera[1],
        -points_camera[2],
        points_camera[0]])
    
    # Convert from 3D to 2D.
    points_2d = np.dot(camera_K, points_camera_opencv)
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]])

    # Discard out of frame points.
    points_2d = points_2d.T
    canvas_mask = \
        (points_2d[:, 0] >= 0.0) & (points_2d[:, 0] < W) & \
        (points_2d[:, 1] >= 0.0) & (points_2d[:, 1] < H) & \
        (points_2d[:, 2] > 0.0)
    u_coords = points_2d[:, 0].astype(np.int)  # Column index.
    v_coords = points_2d[:, 1].astype(np.int)  # Row index.
    
    # Flatten arrays to allow selection of indices in one dimension.
    uv_coords = u_coords + v_coords * W
    rgb_image_flat = rgb_image.reshape((H * W, 3))
    
    # Assign RGB values in [0, 1] where valid, -1 otherwise.
    pcl_rgb = np.ones((N, 3), dtype=lidar_pcl.dtype) * (-1.0)
    pcl_rgb[canvas_mask] = rgb_image_flat[uv_coords[canvas_mask]] / 255.0

    # Combine all information into the final point cloud.
    colorized_pcl = np.zeros((N, D + 3), dtype=lidar_pcl.dtype)
    colorized_pcl[..., :D] = lidar_pcl
    colorized_pcl[..., D:] = pcl_rgb

    return colorized_pcl


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
    result = pcl[mask_xyz]
    return result


def filter_pcl_bounds_carla_input_numpy(pcl, min_z=-1.0, other_bounds=20.0, cube_mode=4):
    '''
    From pcl geometry.py.
    '''
    if cube_mode == 1:
        # NOTE: x > -8 to allow for context.
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=-other_bounds * 0.5, x_max=other_bounds * 2.0, y_min=-other_bounds * 1.0,
            y_max=other_bounds * 1.0, z_min=min_z, z_max=other_bounds * 0.5)

    elif cube_mode == 2:
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=-other_bounds * 0.6, x_max=other_bounds * 2.4, y_min=-other_bounds * 0.8,
            y_max=other_bounds * 0.8, z_min=min_z, z_max=other_bounds * 0.6)

    elif cube_mode == 3:
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=-other_bounds * 0.7, x_max=other_bounds * 2.2, y_min=-other_bounds * 1.0,
            y_max=other_bounds * 1.0, z_min=min_z, z_max=other_bounds * 0.5)

    elif cube_mode == 4:
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=-other_bounds * 0.7, x_max=other_bounds * 2.5, y_min=-other_bounds * 1.0,
            y_max=other_bounds * 1.0, z_min=min_z, z_max=other_bounds * 0.5)

    return pcl


def filter_pcl_bounds_carla_output_numpy(pcl, min_z=-1.0, other_bounds=16.0, cube_mode=4):
    '''
    Similar to pcl geometry.py.
    '''
    padding = 0.0

    # NOTE: x > 0.0 because this is the output cube!
    if cube_mode == 1:
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=0.0 - padding, x_max=other_bounds * 2.0 + padding,
            y_min=-other_bounds * 1.0 - padding, y_max=other_bounds * 1.0 + padding,
            z_min=min_z, z_max=other_bounds * 0.5 + padding)

    elif cube_mode == 2:
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=0.0 - padding, x_max=other_bounds * 2.4 + padding,
            y_min=-other_bounds * 0.8 - padding, y_max=other_bounds * 0.8 + padding,
            z_min=min_z, z_max=other_bounds * 0.4 + padding)

    elif cube_mode == 3:
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=0.0 - padding, x_max=other_bounds * 2.2 + padding,
            y_min=-other_bounds * 1.0 - padding, y_max=other_bounds * 1.0 + padding,
            z_min=min_z, z_max=other_bounds * 0.4 + padding)

    elif cube_mode == 4:
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=0.0 - padding, x_max=other_bounds * 2.5 + padding,
            y_min=-other_bounds * 1.0 - padding, y_max=other_bounds * 1.0 + padding,
            z_min=min_z, z_max=other_bounds * 0.4 + padding)
    
    return pcl


def has_moved(old_transform, new_transform, threshold=1.0):
    delta = abs(old_transform.location.x - new_transform.location.x) + \
        abs(old_transform.location.y - new_transform.location.y) + \
            abs(old_transform.location.z - new_transform.location.z)
    return delta >= threshold
