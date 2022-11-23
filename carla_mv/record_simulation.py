'''
Records a CARLA simulation where a bunch of vehicles and pedestrians are going about their daily
lives. This is done in synchronized mode such that it can later be played back deterministically in
exactly the same way.
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
import random
import sys
import time
import tqdm

# Internal imports.
import my_utils
from my_utils import str2bool


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--host', default='127.0.0.1', type=str,
                        help='IP address of the host server.')
    parser.add_argument('--port', default=2000, type=int,
                        help='TCP port to connect to.')
    parser.add_argument('--seed', type=int,
                        help='Random device seed if determinism is desired for Traffic Manager (optional).')
    parser.add_argument('--record_path', type=str,
                        help='File path to write the recorded simulation.')
    parser.add_argument('--ignore_if_exist', default=True, type=str2bool,
                        help='Halt if the output files already exist.')
    parser.add_argument('--world', default='Town10HD_Opt', type=str,
                        help='Map / world name to load.')
    parser.add_argument('--num_vehicles', default=120, type=int,
                        help='Number of vehicles.')
    parser.add_argument('--num_walkers', default=120, type=int,
                        help='Number of pedestrians / walkers.')
    parser.add_argument('--safe', default=True, type=str2bool,
                        help='Avoid spawning vehicles prone to accidents.')
    parser.add_argument('--four_wheels', default=False, type=str2bool,
                        help='Avoid spawning bikes, motorcycles, etc.')
    parser.add_argument('--no_rendering', default=False, type=str2bool,
                        help='Activate no rendering mode.')
    parser.add_argument('--num_frames', default=2000, type=int,
                        help='Number of ticks to record.')
    parser.add_argument('--tick_offset', default=600, type=int,
                        help='Start recording after this number of ticks after the desired number of actors have been spawned, to allow for a regime situation to settle.')
    parser.add_argument('--fps', default=10, type=float,
                        help='Frames per second in the simulated world. The simulated duration is then num_frames * fps.')

    args = parser.parse_args()
    
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    if args.ignore_if_exist and os.path.exists(args.record_path) and \
            os.path.isfile(args.record_path) and \
            os.path.getsize(args.record_path) >= 2 * 1024 * 1024:
        print(f'Record file path {args.record_path} already exists and is at least 2 MB!')
        print('Exiting...')
        sys.exit(0)

    # Setup logging and output directory.
    output_parent = str(pathlib.Path(args.record_path).parent)
    os.makedirs(output_parent, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Save arguments.
    my_utils.save_args(args, os.path.join(output_parent, 'record_args.txt'))

    # Instantiate client and load world.
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    all_vehicle_ids = []  # List of int.
    all_walker_ids = []  # List of dict from str to int.
    all_ids = []  # List of int.
    traffic_manager = None

    try:

        # -----
        # Setup
        # -----
        world = client.get_world()
        map_name = world.get_map().name.split('/')[-1]
        if map_name != args.world:
            print('Loading map:', args.world)
            world = client.load_world(args.world)
        else:
            print('Map', args.world, 'already loaded.')

        # Instantiate traffic manager.
        traffic_manager = client.get_trafficmanager(args.port + 1234)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        # Apply synchronicity settings.
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        my_utils.destroy_all_dynamic_actors(client, world)

        blueprints_all_ids = sorted([x.id for x in list(world.get_blueprint_library())])
        blueprints_vehicles = my_utils.get_actor_blueprints(world, 'vehicle.*', 'All')
        blueprints_walkers = my_utils.get_actor_blueprints(world, 'walker.pedestrian.*', '2')

        if args.four_wheels:
            blueprints_vehicles = [x for x in blueprints_vehicles if int(
                x.get_attribute('number_of_wheels')) == 4]
        if args.safe:
            blueprints_vehicles = [x for x in blueprints_vehicles if not x.id.endswith('microlino')]
            blueprints_vehicles = [x for x in blueprints_vehicles if not x.id.endswith('carlacola')]
            blueprints_vehicles = [
                x for x in blueprints_vehicles if not x.id.endswith('cybertruck')]
            blueprints_vehicles = [x for x in blueprints_vehicles if not x.id.endswith('t2')]
            blueprints_vehicles = [x for x in blueprints_vehicles if not x.id.endswith('sprinter')]
            blueprints_vehicles = [x for x in blueprints_vehicles if not x.id.endswith('firetruck')]
            blueprints_vehicles = [x for x in blueprints_vehicles if not x.id.endswith('ambulance')]

        vehicle_spawn_points = world.get_map().get_spawn_points()
        random.shuffle(vehicle_spawn_points)
        print('len(vehicle_spawn_points):', len(vehicle_spawn_points))
        car_lights_on = True

        FutureActor = carla.command.FutureActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        SpawnActor = carla.command.SpawnActor
        VehicleLightState = carla.VehicleLightState

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for i in range(args.num_vehicles):

            # Configure car attributes.
            blueprint = random.choice(blueprints_vehicles)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # Prepare the light state of the cars to spawn.
            if car_lights_on:
                light_state = VehicleLightState.Position | VehicleLightState.LowBeam
            else:
                light_state = VehicleLightState.NONE

            # Spawn the cars, set their autopilot, light state, and position.
            transform = vehicle_spawn_points[i % len(vehicle_spawn_points)]
            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                         .then(SetVehicleLightState(FutureActor, light_state)))

        for response in client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                all_vehicle_ids.append(response.actor_id)
                all_ids.append(response.actor_id)

        # Configure traffic manager, which includes creating some dangerous driving.
        all_vehicle_actors = world.get_actors(all_vehicle_ids)
        spacing = np.random.uniform(3.0, 4.0)
        traffic_manager.set_global_distance_to_leading_vehicle(spacing)
        traffic_manager.global_percentage_speed_difference(30.0)
        for vehicle in all_vehicle_actors:
            if np.random.rand() < 0.4:
                traffic_manager.ignore_lights_percentage(vehicle, 10.0)
                traffic_manager.ignore_signs_percentage(vehicle, 10.0)

        # -------------
        # Spawn walkers
        # -------------
        runningProb = 0.1
        crossingProb = 0.1
        world.set_pedestrians_cross_factor(crossingProb)

        # Retrieve all the possible spawn locations.
        walker_spawn_points = []
        for i in range(args.num_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                loc.x += np.random.rand() * 4.0 - 2.0
                loc.y += np.random.rand() * 4.0 - 2.0
                spawn_point.location = loc
                walker_spawn_points.append(spawn_point)
        random.shuffle(walker_spawn_points)

        # Spawn walkers.
        batch = []
        walker_speed = []
        for spawn_point in walker_spawn_points:
            walker_bp = random.choice(blueprints_walkers)

            # Set speed.
            if walker_bp.has_attribute('speed'):
                if random.random() <= runningProb:
                    # Running.
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                    # Walking.
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                print('Walker has no speed')
                walker_speed.append(0.0)

            batch.append(SpawnActor(walker_bp, spawn_point))

        # Log errors and update walker speeds.
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                all_walker_ids.append({'id': results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        # Spawn walker controllers and link walker IDs with their controller IDs.
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(all_walker_ids)):
            batch.append(SpawnActor(walker_controller_bp,
                                    carla.Transform(), all_walker_ids[i]['id']))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                all_walker_ids[i]['con'] = results[i].actor_id
        for i in range(len(all_walker_ids)):
            all_ids.append(all_walker_ids[i]['con'])
            all_ids.append(all_walker_ids[i]['id'])

        # Tick to ensure the client receives the latest transforms of the walkers we just created.
        world.tick()

        # Initialize controllers and set targets to walk to.
        for i, walker_id_pair in enumerate(all_walker_ids):
            walker_actor = world.get_actor(walker_id_pair['con'])
            walker_actor.start()
            walker_actor.go_to_location(world.get_random_location_from_navigation())
            walker_actor.set_max_speed(float(walker_speed[i]))

        print()
        print(f'Spawned {len(all_vehicle_ids)} vehicles and {len(all_walker_ids)} walkers.')
        print('Press Ctrl+C to exit early.')
        print()

        # ---------
        # Recording
        # ---------

        # Set spectator pose.
        vehicle_focus = my_utils.find_kth_vehicle(world, k=0)
        print('vehicle_focus.id:', vehicle_focus.id)
        transform = vehicle_focus.get_transform()
        transform.location.z += 3.0
        world.get_spectator().set_transform(transform)

        # Pre-regime.
        for tick_idx in tqdm.tqdm(range(args.tick_offset)):
            world.tick()

        # Wait for motion of focus vehicle.
        print()
        print('Waiting for focus vehicle (k=0) to move...')
        ref_tf = vehicle_focus.get_transform()
        wait_ticks = 0
        while True:
            world.tick()
            wait_ticks += 1
            cur_tf = vehicle_focus.get_transform()
            if my_utils.has_moved(ref_tf, cur_tf, threshold=2.0):
                print('ref_tf:', ref_tf)
                print('cur_tf:', cur_tf)
                break
        print('Waited for number of ticks:', wait_ticks)
        print()

        # Start recording.
        print('Start recording to file:', client.start_recorder(args.record_path, False))

        # Regime loop.
        for tick_idx in tqdm.tqdm(range(args.num_frames)):
            world.tick()

    except Exception as e:
        print(e)

    finally:

        print('Stop recording')
        client.stop_recorder()

        if world is not None:
            # Disable synchronous mode.
            settings = world.get_settings()
            if traffic_manager is not None:
                traffic_manager.set_synchronous_mode(False)
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

            # Stop walker AI controllers.
            for walker_id_pair in all_walker_ids:
                controller = world.get_actor(walker_id_pair['con'])
                controller.stop()

            # print('Destroying all vehicles and walkers...')
            # my_utils.destroy_by_ids(client, all_ids)
            my_utils.destroy_all_dynamic_actors(client, world)

        time.sleep(0.1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print()
        print('Done!')
