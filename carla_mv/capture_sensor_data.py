'''
Plays back a recorded CARLA simulation multiple times, and gathers useful sensor data for each run.
This eventually gives rise to multiple 'views' of the same scene.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields (CVPR 2022).
'''

# Library imports.
import argparse
import carla
import glob
import json
import logging
import numpy as np
import os
import pathlib
import queue
import random
import sys
import time
import tqdm
import traceback

# Internal imports.
import my_utils
from my_utils import str2bool


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--host', default='127.0.0.1', type=str,
                        help='IP address of the host server.')
    parser.add_argument('--port', default=2000, type=int,
                        help='TCP port to connect to.')
    parser.add_argument('--record_path', type=str,
                        help='File path to write the recorded simulation.')
    parser.add_argument('--ignore_if_exist', default=True, type=str2bool,
                        help='Halt if the output files already exist.')
    parser.add_argument('--no_rendering', default=False, type=str2bool,
                        help='Activate no rendering mode.')
    parser.add_argument('--num_frames', default=1010, type=int,
                        help='Number of ticks to playback.')
    parser.add_argument('--fps', default=10, type=float,
                        help='Frames per second in the recorded simulated world.')
    parser.add_argument('--time_factor', default=1.0, type=float,
                        help='Playback time factor. The effective FPS experienced by sensors will be = fps / time_factor.')
    parser.add_argument('--capture_path', type=str,
                        help='Directory path to write the captured sensor data to.')
    parser.add_argument('--follow_vehicle_index', default=0, type=float,
                        help='Which vehicle, ordered by actor ID, to focus on (set 0 for the first, 1 for the second, etc.).')
    parser.add_argument('--weather_preset', default=6, type=int,
                        help='Weather conditions from a predefined list of noon / sunset / night scenarios.')
    parser.add_argument('--seed', default=1234, type=int,
                        help='Random seed for magic sensor pose noise.')
    parser.add_argument('--weather_preset_name', default='', type=str,
                        help='Auto filled argument, do not modify.')

    args = parser.parse_args()
    return args


class CarlaSyncMode(object):
    '''
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    '''

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 30)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def save_rgb_image(image, dst_fp):
    image.save_to_disk(dst_fp, carla.ColorConverter.Raw)


def save_depth_image(image, dst_fp):
    image.save_to_disk(dst_fp, carla.ColorConverter.Depth)


def save_segm_image(image, dst_fp):
    image.save_to_disk(dst_fp, carla.ColorConverter.CityScapesPalette)


def save_lidar_pcl(lidar_measurement, dst_fp, segm=False, save=True):
    if segm:
        pcl_float = np.frombuffer(lidar_measurement.raw_data, dtype=np.dtype('f4'))
        pcl_float = np.reshape(pcl_float, (int(pcl_float.shape[0] / 6), 6))
        pcl_float = pcl_float[..., :4]  # (N, 4) with (x, y, z, cosine_angle).
        pcl_int = np.frombuffer(lidar_measurement.raw_data, dtype=np.dtype('u4'))
        pcl_int = np.reshape(pcl_int, (int(pcl_int.shape[0] / 6), 6))
        pcl_int = pcl_int[..., 4:6]  # (N, 2) with (object_index, semantic_tag).
        pcl_int = pcl_int.astype(np.float32)
        pcl = np.concatenate([pcl_float, pcl_int], axis=1)  # (N, 6).
        # Last two columns are now readable.

    else:
        pcl = np.frombuffer(lidar_measurement.raw_data, dtype=np.dtype('f4'))
        pcl = np.reshape(pcl, (int(pcl.shape[0] / 4), 4))  # (N, 4) with (x, y, z, intensity).

    if save:
        np.save(dst_fp, pcl)
    else:
        return pcl


def colorize_save_lidar_pcl(
        lidar_measurement, rgb_image, lidar_to_world, world_to_camera,
        camera_K, dst_fp, segm=False):
    lidar_pcl = save_lidar_pcl(lidar_measurement, None, segm=segm, save=False)
    (H, W) = rgb_image.height, rgb_image.width
    rgb_image = np.asarray(rgb_image.raw_data)
    rgb_image = rgb_image.reshape((H, W, 4))
    rgb_image = rgb_image[..., :3][..., ::-1]  # BGRA -> BGR -> RGB.
    colorized_pcl = my_utils.colorize_lidar_pcl(
        lidar_pcl, rgb_image, lidar_to_world, world_to_camera, camera_K)
    np.save(dst_fp, colorized_pcl)


def apply_transform_noise(transform, loc_noise, rot_noise):
    transform.location.x += (np.random.rand() * 2.0 - 1.0) * loc_noise
    transform.location.y += (np.random.rand() * 2.0 - 1.0) * loc_noise
    transform.location.z += (np.random.rand() * 2.0 - 1.0) * loc_noise
    transform.rotation.pitch += (np.random.rand() * 2.0 - 1.0) * rot_noise
    transform.rotation.yaw += (np.random.rand() * 2.0 - 1.0) * rot_noise
    transform.rotation.roll += (np.random.rand() * 2.0 - 1.0) * rot_noise
    return transform


def get_magic_transforms(args, offset_x, offset_z, loc_noise=0.0, rot_noise=0.0):
    # delta_forward = 16.0
    # delta_side = 8.0
    # delta_up = 4.0
    # magic_left_transform = carla.Transform(
    #     carla.Location(x=delta_forward + offset_x, y=-delta_side, z=delta_up + offset_z),
    #     carla.Rotation(pitch=-30.0, yaw=120.0))
    # magic_right_transform = carla.Transform(
    #     carla.Location(x=delta_forward + offset_x, y=delta_side, z=delta_up + offset_z),
    #     carla.Rotation(pitch=-30.0, yaw=-120.0))
    # delta_forward = 20.0
    # delta_up = 20.0
    # magic_top_transform = carla.Transform(
    #     carla.Location(x=delta_forward + offset_x, y=0.0, z=delta_up + offset_z),
    #     carla.Rotation(pitch=-90.0, yaw=-90.0))

    delta_forward = 20.0
    delta_side = 8.0
    delta_up = 4.0
    magic_left_transform = carla.Transform(
        carla.Location(x=delta_forward + offset_x, y=-delta_side, z=delta_up + offset_z),
        carla.Rotation(pitch=-30.0, yaw=105.0))
    magic_right_transform = carla.Transform(
        carla.Location(x=delta_forward + offset_x, y=delta_side, z=delta_up + offset_z),
        carla.Rotation(pitch=-30.0, yaw=-105.0))
    delta_forward = 20.0
    delta_up = 14.0
    magic_top_transform = carla.Transform(
        carla.Location(x=delta_forward + offset_x, y=0.0, z=delta_up + offset_z),
        carla.Rotation(pitch=-90.0, yaw=-90.0))

    if loc_noise > 0.0 or rot_noise > 0.0:
        magic_left_transform = apply_transform_noise(magic_left_transform, loc_noise, rot_noise)
        magic_right_transform = apply_transform_noise(magic_right_transform, loc_noise, rot_noise)
        magic_top_transform = apply_transform_noise(magic_top_transform, loc_noise, rot_noise)

    return (magic_left_transform, magic_right_transform, magic_top_transform)


def get_camera_blueprint(args, world, pattern, is_forward):
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find(pattern)

    blueprint.set_attribute('image_size_x', '800')
    blueprint.set_attribute('image_size_y', '600')
    blueprint.set_attribute('fov', '120.0')
    
    return blueprint


def get_lidar_blueprint(args, world, pattern, lidar_mode):
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find(pattern)

    if lidar_mode == 'input':
        blueprint.set_attribute('range', '48.0')
        blueprint.set_attribute('upper_fov', '10.0')
        blueprint.set_attribute('lower_fov', '-30.0')

        # Used by model, but resolution / quality should be sufficiently high.
        # Vertical angle step = 40 / 128 = 0.31 deg.
        # Horizontal angle step = 120 / (448000 / 10 / 128) = 0.34 deg.
        blueprint.set_attribute('channels', '128')
        blueprint.set_attribute('points_per_second', '448000')
        blueprint.set_attribute('horizontal_fov', '120.0')

        # Used by model.
        # Vertical angle step = 40 / 64 = 0.63 deg.
        # Horizontal angle step = 120 / (112000 / 10 / 64) = 0.69 deg.
        # blueprint.set_attribute('channels', '64')
        # blueprint.set_attribute('points_per_second', '112000')
        # blueprint.set_attribute('horizontal_fov', '120.0')

        if not('semantic' in pattern):
            blueprint.set_attribute('dropoff_general_rate', '0.0')
            blueprint.set_attribute('noise_stddev', '0.0')

    elif lidar_mode == 'target':
    
        # Used as supervision for continuous representation => denser data.
        # Vertical angle step = 80 / 256 = 0.31 deg.
        # Horizontal angle step = 120 / (112000 / 10 / 64) = 0.34 deg.
        blueprint.set_attribute('channels', '256')
        blueprint.set_attribute('points_per_second', '896000')
        blueprint.set_attribute('range', '48.0')
        blueprint.set_attribute('upper_fov', '40.0')
        blueprint.set_attribute('lower_fov', '-40.0')
        blueprint.set_attribute('horizontal_fov', '120.0')

        if not('semantic' in pattern):
            blueprint.set_attribute('dropoff_general_rate', '0.0')
            blueprint.set_attribute('noise_stddev', '0.0')

    else:
        raise ValueError()

    return blueprint


def spawn_sensors(args, world, vehicle_focus, lidar_mode):
    sensors = []
    sensor_names = []
    offset_x = vehicle_focus.bounding_box.extent.x + 1e-2  # Half the vehicle length.
    offset_z = 1.0
    forward_transform = carla.Transform(carla.Location(x=offset_x, z=offset_z),
                                        carla.Rotation(yaw=0.0))

    # Forward-facing RGB camera.
    # https://carla.readthedocs.io/en/0.9.12/ref_sensors/#camera-lens-distortion-attributes_1
    blueprint = get_camera_blueprint(args, world, 'sensor.camera.rgb', True)
    camera_rgb = world.spawn_actor(blueprint, forward_transform, attach_to=vehicle_focus)
    sensors.append(camera_rgb)
    sensor_names.append('forward_rgb')

    # Get intrinsic camera parameters for later LiDAR alignment.
    image_width = blueprint.get_attribute('image_size_x').as_int()
    image_height = blueprint.get_attribute('image_size_y').as_int()
    image_fov = blueprint.get_attribute('fov').as_float()
    focal_length = image_width / (2.0 * np.tan(image_fov * np.pi / 360.0))
    camera_K = np.identity(3)
    camera_K[0, 0] = camera_K[1, 1] = focal_length
    camera_K[0, 2] = image_width / 2.0
    camera_K[1, 2] = image_height / 2.0

    # Two LiDAR sensors (regular and semantic).
    print('lidar_mode:', lidar_mode)
    blueprint = get_lidar_blueprint(args, world, 'sensor.lidar.ray_cast', 'input')
    lidar = world.spawn_actor(blueprint, forward_transform, attach_to=vehicle_focus)
    sensors.append(lidar)
    sensor_names.append('forward_lidar')

    blueprint = get_lidar_blueprint(args, world, 'sensor.lidar.ray_cast_semantic', 'input')
    lidar = world.spawn_actor(blueprint, forward_transform, attach_to=vehicle_focus)
    sensors.append(lidar)
    sensor_names.append('forward_lidar_segm')

    loc_noise = 1.0
    rot_noise = 2.0
    # DEBUG:
    # loc_noise = 5.0
    # rot_noise = 20.0
    # loc_noise = 0.0
    # rot_noise = 0.0
    magic_transforms = get_magic_transforms(args, offset_x, offset_z,
                                            loc_noise=loc_noise, rot_noise=rot_noise)

    # Two RGB cameras in magic positions.
    blueprint = get_camera_blueprint(args, world, 'sensor.camera.rgb', False)
    camera_rgb = world.spawn_actor(
        blueprint, magic_transforms[0], attach_to=vehicle_focus)
    sensors.append(camera_rgb)
    sensor_names.append('magic_left_rgb')
    camera_rgb = world.spawn_actor(
        blueprint, magic_transforms[1], attach_to=vehicle_focus)
    sensors.append(camera_rgb)
    sensor_names.append('magic_right_rgb')
    camera_rgb = world.spawn_actor(
        blueprint, magic_transforms[2], attach_to=vehicle_focus)
    sensors.append(camera_rgb)
    sensor_names.append('magic_top_rgb')

    # Two semantic LiDAR sensors in magic positions.
    blueprint = get_lidar_blueprint(args, world, 'sensor.lidar.ray_cast_semantic', 'target')
    lidar_segm = world.spawn_actor(
        blueprint, magic_transforms[0], attach_to=vehicle_focus)
    sensors.append(lidar_segm)
    sensor_names.append('magic_left_lidar_segm')
    lidar_segm = world.spawn_actor(
        blueprint, magic_transforms[1], attach_to=vehicle_focus)
    sensors.append(lidar_segm)
    sensor_names.append('magic_right_lidar_segm')
    lidar_segm = world.spawn_actor(
        blueprint, magic_transforms[2], attach_to=vehicle_focus)
    sensors.append(lidar_segm)
    sensor_names.append('magic_top_lidar_segm')

    return (sensors, sensor_names, camera_K)


def process_sensor_data(args, data, camera_K, sensors, sensor_names, output_path, tick_idx):
    (image_rgb, lidar_data, lidar_segm_data,
        image_rgb_magic_left, image_rgb_magic_right, image_rgb_magic_top,
        lidar_segm_data_magic_left, lidar_segm_data_magic_right,
        lidar_segm_data_magic_top) = data
    (forward_tf, magic_left_tf, magic_right_tf, magic_top_tf) = \
        (sensors[0].get_transform(), sensors[3].get_transform(),
            sensors[4].get_transform(), sensors[5].get_transform())

    dst_fp = os.path.join(output_path, f'{tick_idx:05d}_forward_rgb.png')
    save_rgb_image(image_rgb, dst_fp)

    dst_fp = os.path.join(output_path, f'{tick_idx:05d}_forward_lidar.npy')
    lidar_to_world = np.array(forward_tf.get_matrix())
    world_to_camera = np.array(forward_tf.get_inverse_matrix())
    colorize_save_lidar_pcl(
        lidar_data, image_rgb, lidar_to_world, world_to_camera, camera_K, dst_fp, segm=False)

    dst_fp = os.path.join(output_path, f'{tick_idx:05d}_forward_lidar_segm.npy')
    colorize_save_lidar_pcl(
        lidar_segm_data, image_rgb, lidar_to_world, world_to_camera,
        camera_K, dst_fp, segm=True)

    dst_fp = os.path.join(output_path, f'{tick_idx:05d}_magic_left_rgb.png')
    save_rgb_image(image_rgb_magic_left, dst_fp)
    dst_fp = os.path.join(output_path, f'{tick_idx:05d}_magic_right_rgb.png')
    save_rgb_image(image_rgb_magic_right, dst_fp)
    dst_fp = os.path.join(output_path, f'{tick_idx:05d}_magic_top_rgb.png')
    save_rgb_image(image_rgb_magic_top, dst_fp)

    dst_fp = os.path.join(output_path, f'{tick_idx:05d}_magic_left_lidar_segm.npy')
    lidar_to_world = np.array(magic_left_tf.get_matrix())
    world_to_camera = np.array(magic_left_tf.get_inverse_matrix())
    colorize_save_lidar_pcl(
        lidar_segm_data_magic_left, image_rgb_magic_left,
        lidar_to_world, world_to_camera, camera_K, dst_fp, segm=True)

    dst_fp = os.path.join(output_path, f'{tick_idx:05d}_magic_right_lidar_segm.npy')
    lidar_to_world = np.array(magic_right_tf.get_matrix())
    world_to_camera = np.array(magic_right_tf.get_inverse_matrix())
    colorize_save_lidar_pcl(
        lidar_segm_data_magic_right, image_rgb_magic_right,
        lidar_to_world, world_to_camera, camera_K, dst_fp, segm=True)

    dst_fp = os.path.join(output_path, f'{tick_idx:05d}_magic_top_lidar_segm.npy')
    lidar_to_world = np.array(magic_top_tf.get_matrix())
    world_to_camera = np.array(magic_top_tf.get_inverse_matrix())
    colorize_save_lidar_pcl(
        lidar_segm_data_magic_top, image_rgb_magic_top,
        lidar_to_world, world_to_camera, camera_K, dst_fp, segm=True)


def capture_run(args, client, output_path, lidar_mode):
    try:

        random.seed(args.seed)
        np.random.seed(args.seed)

        world = client.get_world()

        # Enable synchronous mode.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        # Start playback.
        client.set_replayer_time_factor(args.time_factor)
        print(client.replay_file(args.record_path, 0.0, 0.0, 0, False))
        time.sleep(3.5)  # Allow for map to be loaded.

        # Set weather (index 6 = Default, count = 22).
        weather_presets = my_utils.find_weather_presets()
        weather_preset_name = weather_presets[args.weather_preset][1]
        print('Weather preset name:', weather_preset_name)
        world.set_weather(weather_presets[args.weather_preset][0])

        # This should spawn all recorded actors.
        world.tick()

        # Find car of interest.
        vehicle_focus = my_utils.find_kth_vehicle(world, args.follow_vehicle_index)
        print('vehicle_focus.id:', vehicle_focus.id)
        all_vehped = my_utils.get_all_vehicles_pedestrians(world)

        (sensors, sensor_names, camera_K) = spawn_sensors(args, world, vehicle_focus, lidar_mode)
        print('sensor_names:', sensor_names)
        # print('camera_K:', camera_K)
        assert len(sensors) == len(sensor_names)

        all_poses = []
        all_matrices = []
        all_bboxes = dict()

        # Loop.
        with CarlaSyncMode(world, *sensors, fps=args.fps) as sync_mode:
            for tick_idx in tqdm.tqdm(range(args.num_frames)):

                # Advance the simulation and wait for sensor data.
                (snapshot, *data) = sync_mode.tick(timeout=10.0)

                # Ensure the sensor (vehicle) pose is horizontally flat at every frame.
                transform = vehicle_focus.get_transform()
                transform.rotation.pitch = 0.0
                transform.rotation.roll = 0.0
                vehicle_focus.set_transform(transform)
                vehicle_focus.set_simulate_physics(False)

                # Convert and save sensor data.
                process_sensor_data(
                    args, data, camera_K, sensors, sensor_names, output_path, tick_idx)

                # Save sensor trajectory.
                cur_poses = []
                cur_matrices = []

                for (sensor_measurement, sensor_name) in zip(data, sensor_names):
                    location = sensor_measurement.transform.location
                    rotation = sensor_measurement.transform.rotation
                    cur_pose = np.array([location.x, location.y, location.z,
                                        rotation.pitch, rotation.roll, rotation.yaw])
                    cur_matrix = np.array(sensor_measurement.transform.get_matrix())
                    cur_poses.append(cur_pose)
                    cur_matrices.append(cur_matrix)

                cur_poses = np.stack(cur_poses)
                cur_matrices = np.stack(cur_matrices)
                all_poses.append(cur_poses)
                all_matrices.append(cur_matrices)

                # Save bounding boxes.
                cur_bboxes = dict()
                for actor in all_vehped:
                    cur_bboxes[actor.id] = dict()
                    for attr in actor.attributes.keys():
                        cur_bboxes[actor.id]['attr_' + attr] = actor.attributes[attr]
                    cur_bboxes[actor.id]['type_id'] = actor.type_id
                    cur_bboxes[actor.id]['extent_x'] = actor.bounding_box.extent.x
                    cur_bboxes[actor.id]['extent_y'] = actor.bounding_box.extent.y
                    cur_bboxes[actor.id]['extent_z'] = actor.bounding_box.extent.z
                    cur_bboxes[actor.id]['location_x'] = actor.bounding_box.location.x
                    cur_bboxes[actor.id]['location_y'] = actor.bounding_box.location.y
                    cur_bboxes[actor.id]['location_z'] = actor.bounding_box.location.z
                    cur_bboxes[actor.id]['rotation_x'] = actor.bounding_box.rotation.pitch
                    cur_bboxes[actor.id]['rotation_y'] = actor.bounding_box.rotation.roll
                    cur_bboxes[actor.id]['rotation_z'] = actor.bounding_box.rotation.yaw
                all_bboxes[tick_idx] = cur_bboxes

        # Stop and destroy sensors.
        for sensor in sensors:
            sensor.stop()
            sensor.destroy()

        all_poses = np.stack(all_poses)
        all_matrices = np.stack(all_matrices)

        dst_fp = os.path.join(output_path, f'sensor_names.txt')
        np.savetxt(dst_fp, sensor_names, fmt='%s')
        dst_fp = os.path.join(output_path, f'sensor_poses.npy')
        np.save(dst_fp, all_poses)
        dst_fp = os.path.join(output_path, f'sensor_matrices.npy')
        np.save(dst_fp, all_matrices)
        dst_fp = os.path.join(output_path, f'camera_K.npy')
        np.save(dst_fp, camera_K)
        dst_fp = os.path.join(output_path, f'bounding_boxes.json')
        with open(dst_fp, 'w') as f:
            json.dump(all_bboxes, f, indent=4)

        return True

    except Exception as e:
        print(e)
        traceback.print_exc()

        return False

    finally:

        # Disable synchronous mode.
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        my_utils.destroy_all_dynamic_actors(client, world)

        time.sleep(0.1)


def main():

    args = get_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    if args.capture_path is None:
        args.capture_path = str(pathlib.Path(args.record_path).parent)
        print('Set capture_path:', args.capture_path)

    if not os.path.exists(args.record_path):
        print(f'Recording {args.record_path} does not exist!')
        print('Exiting...')
        sys.exit(1)

    client_initialized = False

    output_dn = 'mv_raw_all'
    output_path = os.path.join(args.capture_path, output_dn)

    # Save weather preset name.
    weather_presets = my_utils.find_weather_presets()
    weather_preset_name = weather_presets[args.weather_preset][1]
    args.weather_preset_name = weather_preset_name

    # Save arguments.
    my_utils.save_args(args, output_path + '_capture_args.txt')

    # Check for last generated file.
    sensor_matrices_fp = os.path.join(output_path, f'sensor_matrices.npy')
    if args.ignore_if_exist and os.path.exists(sensor_matrices_fp):
        print(f'Capture directory path {output_path} already exists and contains '
                'sensor_matrices.npy!')
        print('Exiting...')
        sys.exit(0)

    # Instantiate client and load world.
    if not client_initialized:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        world = client.get_world()
        my_utils.destroy_all_dynamic_actors(client, world)
        client_initialized = True

    # Capture sensor data.
    lidar_mode = 'multiview'
    result = capture_run(args, client, output_path, lidar_mode)

    if not result:
        sys.exit(1)


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print()
        print('Done!')
