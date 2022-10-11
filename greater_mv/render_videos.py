# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
'''
Modified by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields (CVPR 2022).
'''

from __future__ import print_function
import traceback
import shutil
import platform
import itertools
import logging
import errno
import numpy as np
from datetime import datetime as dt
import json
import argparse
import colorsys
import random
import math
import os
import pathlib
import sys
import time
import tqdm
import faulthandler; faulthandler.enable()
from contextlib import contextmanager

sys.path.append(os.getcwd())

_INSIDE_BLENDER = True
_RUNNING_ON_SERVER = True


# For compatibility when running via Blender; imports might not work otherwise.
# _BASE_DIR = r'/path/to/this/repo'
# sys.path.append(_BASE_DIR)

from movement_record import MovementRecord  # nopep8

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError as e:
    _INSIDE_BLENDER = False
if _INSIDE_BLENDER:
    try:
        import utils
        import actions
    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import "
              "utils.py. You may need to add a .pth file to the site-packages "
              "of Blender's bundled python with a command like this:\n "
              "echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth"  # noQA
              "\nWhere $BLENDER is the directory where Blender is installed, "
              "and $VERSION is your Blender version (such as 2.78).")
        sys.exit(1)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


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


parser = argparse.ArgumentParser()


# Input options
parser.add_argument(
    '--base_scene_blendfile',
    # default='data/base_scene.blend',
    default='data/base_scene_withAxes.blend',
    help="Base blender file on which all scenes are based; includes " +
         "ground plane, lights, and camera.")
parser.add_argument(
    '--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the 'materials' and 'shapes' fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument(
    '--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument(
    '--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument(
    '--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument(
    '--min_objects', default=5, type=int,  # CATER default = 5
    help="The minimum number of objects to place in each scene")
parser.add_argument(
    '--max_objects', default=10, type=int,  # CATER default = 10
    help="The maximum number of objects to place in each scene")
parser.add_argument(
    '--min_dist', default=0.1, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument(
    '--margin', default=0.2, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart; making resolving " +
         "spatial relationships slightly less ambiguous.")

# NOTE: This argument appears to be actually unused?
parser.add_argument(
    '--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")

parser.add_argument(
    '--max_retries', default=100, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Custom or modified arguments:
parser.add_argument(
    '--export_png', default=True, type=str2bool,
    help="If True, export individual frames as PNG images; otherwise output as AVI video file. "
    "Output file names will be like 0123.png.")
parser.add_argument(
    '--export_depth', default=True, type=str2bool,
    help="If True, also export depth maps (z-values) using compositor nodes. "
    "Output file names will be like 0123_depth.exr.")
parser.add_argument(
    '--export_flat', default=True, type=str2bool,
    help="If True, also export instance segmentation maps (unique color per object). "
    "Output file names will be like 0123_preflat.png")
parser.add_argument(
    '--scatter_flat', default=True, type=str2bool,
    help='If True, use a separate mask file per object ID, in order to enable X-ray annotations. '
    "Output file names will be like 0123_preflat_obj4.png. NOTE: only done for snitch for speed")
parser.add_argument(
    '--flat_val_test_only', default=False, type=str2bool,
    help='If True, export instance segmentation maps in the validation and test splits only.')
parser.add_argument(
    '--train_val_test_split', default=False, type=str2bool,
    help="If True, organize dataset split according to 0.80 - 0.10 - 0.10.")
parser.add_argument(
    '--static_objects', default=False, type=str2bool,
    help='If True, do not move objects such that this is a fixed 3D scene.')
parser.add_argument(
    '--size_multiplier', default=1.0, type=float,
    help='Make all objects larger (>1.0) or smaller (<1.0), for example to encourage occlusions.')
parser.add_argument(
    '--fully_random_objects', default=False, type=str2bool,
    help="If False, first, second, and third objects are the snitch, medium cone, and large cone respectively."
    "If True, override this CATER tradition and make everything random.")
parser.add_argument(
    '--singular_movement', default=False, type=str2bool,
    help='If False, arbitrarily many objects can move at once. If True, only one object moves at a time.')
parser.add_argument(
    '--num_cameras', default=1, type=int,
    help="Number of different cameras (viewpoints) to render the scene from, i.e. export multiple ground truths if >1.")
parser.add_argument(
    "--random_camera_motion", default=True, type=str2bool,
    help="If True, render the video with random camera motion (otherwise fixed).")
parser.add_argument(
    '--front_back_bird', default=False, type=str2bool,
    help='If True and fixed cameras, force predetermined camera poses '
    '(front, back, bird for view 1, 2, 3 respectively).')
parser.add_argument(
    '--speed_factor', default=1, type=int,
    help='Playback rate of the scene for lower effective frame rate when debugging. '
    'Any value > 1 will speed up everything by the specified integer factor. '
    'Designed to replace FPS which apparently does not actually achieve this.')
parser.add_argument(
    '--random_roll_degrees', default=0.0, type=float,
    help='If > 0, add random roll to camera up to this maximum angle in degrees.')
parser.add_argument(
    '--extra_vertical_movement', default=0.0, type=float,
    help='If > 0, periodically aim the camera from lower positions (Z) and toward higher targets (Z).')
parser.add_argument(
    '--random_horizontal_targets', default=0.0, type=float,
    help='If > 0, aim at points spread over the ground (X, Y) rather than the exact center.')
parser.add_argument(
    '--random_static_cameras', default=False, type=str2bool,
    help='If True, render with fixed but random camera poses, uniformly chosen within certain angle ranges. '
    'extra_vertical_movement and random_horizontal_targets ignored in this case.')
parser.add_argument(
    '--camera_radius', default=15.0, type=float,
    help='If random_static_cameras, deterministic camera distance to the origin.')
parser.add_argument(
    '--any_containment', default=True, type=str2bool,
    help='If False, never have any objects contain each other at all.')
parser.add_argument(
    '--train_containment', default=True, type=str2bool,
    help='If False, never have any objects contain each other in the training split.')
parser.add_argument(
    '--containment_even_only', default=False, type=str2bool,
    help='If True, never have any objects contain each other in every odd-numbered scene index.')
parser.add_argument(
    '--skip_nonempty', default=True, type=str2bool,
    help='If True, skip scene folders that already have files or directories within them. '
    'May be useful for restarting paused jobs.')
parser.add_argument(
    '--vagabond_snitch', default=False, type=str2bool,
    help='If True, force the snitch to move around more often.')

# Output settings
parser.add_argument(
    '--start_index', default=0, type=int,
    help="The inclusive index at which to start for numbering rendered scenes. " +
         "Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument(
    '--stop_index', default=-1, type=int,
    help="The exclusive index at which to stop for numbering rendered scenes.")
parser.add_argument(
    '--num_scenes', default=1, type=int,
    help="The number of scenes to render.")
parser.add_argument(
    '--parallel_mode', action='store_true',
    help="Set if running on multiple nodes/GPUs. Will use lock files "
         "to synchronize.")
parser.add_argument(
    '--filename_prefix', default='GREATER',  # CATER default = 'CATER'
    help="This prefix will be prepended to rendered images and JSON scenes")
parser.add_argument(
    '--output_dir', default='default_output/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument(
    '--save_blendfiles', type=int, default=1,  # set to 1 for testing
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument(
    '--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument(
    '--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument(
    '--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument(
    '--cpu', default=True, type=str2bool,
    help="Setting true disables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "GPU rendering to work. For specifying a GPU, use "
         "CUDA_VISIBLE_DEVICES before running singularity. "
         "I recommend to use GPU only when render_num_samples is (very) high.")
parser.add_argument(
    '--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument(
    '--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument(
    '--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument(
    '--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument(
    '--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument(
    '--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument(
    '--render_num_samples', default=128, type=int,  # CLEVR was 512
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument(
    '--render_min_bounces', default=2, type=int,  # default 8
    help="The minimum number of bounces to use for rendering.")
parser.add_argument(
    '--render_max_bounces', default=2, type=int,  # default 8
    help="The maximum number of bounces to use for rendering.")
parser.add_argument(
    '--render_tile_size', default=40, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")
# render_tile_size = 256 => 1.25 GB per GPU on cv01.
# NOTE: Use CPU on servers, and smaller tile size, much faster!

# Video options
parser.add_argument(
    '--num_frames', default=300, type=int,  # CATER default = 300, changed for debugging
    help="Number of frames to render.")
parser.add_argument(
    '--num_flips', default=10, type=int,
    help="Number of flips to render.")
parser.add_argument(
    '--fps', default=24, type=int,
    help="Video FPS.")
parser.add_argument(
    '--render', default=True, type=str2bool,
    help="Render the video. Otherwise will only store the blend file.")
parser.add_argument(
    "--max_motions",
    help="Number of max objects to move in the single object case. "
         "This ensures the actions are sparser, and random perf lower.",
    type=int, default=999999)

parser.add_argument(
    '-d', '--debug', action='store_true',
    help="Run in debug mode. Will crash on exceptions.")
parser.add_argument(
    '--suppress_blender_logs', action='store_true',
    help="Dont print extra blender logs.")
parser.add_argument(
    "-v", "--verbose", help="increase output verbosity",
    action="store_true")


# Add some more space at the end for animations, objects, cameras moving etc. without glitches.
_SCENE_EXTRA_FRAMES = 30
# _SCENE_EXTRA_FRAMES = 0


def print_sep():
    print('-----------------------------------------------------------------------------------------------------------')


def mkdir_p(path):
    """
    Make all directories in `path`. Ignore errors if a directory exists.
    Equivalent to `mkdir -p` in the command line, hence the name.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def lock(fpath):
    lock_fpath = fpath + '.lock'
    if os.path.exists(fpath) or os.path.exists(lock_fpath):
        return False
    try:
        mkdir_p(lock_fpath)
        return True
    except Exception as e:
        logging.warning('Unable to lock {} due to {}'.format(fpath, e))
        return False


def unlock(fpath):
    lock_fpath = fpath + '.lock'
    try:
        os.rmdir(lock_fpath)
    except Exception as e:
        logging.warning('Maybe some other job already finished {}. Got {}'
                        .format(fpath, e))


def get_split(args, scn_idx):
    split_fraction = scn_idx / args.num_scenes
    if args.train_val_test_split:
        dataset_split = 'train' if split_fraction < 0.80 else 'val' if split_fraction < 0.90 else 'test'
    else:
        dataset_split = ''  # To be compatible with postprocess_dataset.
    return dataset_split


def main(args):

    # Preprocess flags.
    if args.stop_index <= 0:
        args.stop_index = args.num_scenes

    num_digits = 6
    prefix = '%s_' % (args.filename_prefix)
    folder_template = '%s%%0%dd' % (prefix, num_digits)  # e.g., GREATER_000012.

    # First ensure all directories exist (useful for job-level parallelization).
    if args.start_index == 0:
        for i in tqdm.tqdm(range(0, args.num_scenes)):
            # Get path dataset split (train / val / test / nothing).
            dataset_split = get_split(args, i)
            base_dp = os.path.join(args.output_dir, dataset_split, folder_template) % i
            os.makedirs(base_dp, exist_ok=True)
    else:
        print('Sleeping due to non-zero start index...')
        time.sleep(1.0 + args.start_index / 500.0)

    all_scene_paths = []

    # Start generation of whole dataset; loop over all scenes.
    for i in range(args.start_index, args.stop_index):
        random.seed(42 + i)
        np.random.seed(42 + i)
        
        # Get path dataset split (train / val / test / nothing).
        dataset_split = get_split(args, i)
        # Get actual output directories and paths.
        current_images_template = os.path.join(args.output_dir, dataset_split, folder_template, 'images', )
        current_scene_template = os.path.join(args.output_dir, dataset_split, folder_template, 'scene.json')
        current_blend_template = os.path.join(args.output_dir, dataset_split, folder_template, 'blend.blend')
        current_poses_template = os.path.join(args.output_dir, dataset_split, folder_template, 'poses')
        
        # Skip if already filled (doesn't check whether partial).
        current_scene_folder = os.path.join(args.output_dir, dataset_split, folder_template) % i
        if os.path.exists(current_scene_folder) and os.path.isdir(current_scene_folder) and \
                len(os.listdir(current_scene_folder)) >= 2 and args.skip_nonempty:
            print('Current scene directory path:', current_scene_folder)
            print('Items already in directory:', len(os.listdir(current_scene_folder)))
            print('Skipping to next scene...')
            print()
            time.sleep(0.05)
            continue

        image_folder_path = current_images_template % i
        scene_path = current_scene_template % i
        all_scene_paths.append(scene_path)
        if args.save_blendfiles == 1:
            blend_path = current_blend_template % i
        else:
            blend_path = None
        pose_folder_path = current_poses_template % i

        # Render this scene.
        logging.info('Working on {}'.format(image_folder_path))
        num_tries = 0
        while num_tries <= args.max_retries // 2 + 1:
            
            num_objects = random.randint(args.min_objects, args.max_objects)        
            try:
                render_scene(
                    args,
                    num_objects=num_objects,
                    output_index=i,
                    output_image_dir=image_folder_path,
                    output_scene_path=scene_path,
                    output_blend_path=blend_path,
                    output_pose_dir=pose_folder_path,
                    dataset_split=dataset_split)
                break

            except Exception as e:
                if args.debug:
                    # unlock(image_folder_path)
                    # unlock(pose_folder_path)
                    raise e
                logging.warning('Didnt work for {} due to {}. Ignoring for now..'
                                .format(image_folder_path, e))
                traceback.print_exc()
                print('=> num_objects:', num_objects)
                print('=> num_tries so far for this scene index:', num_tries)
                
                time.sleep(0.2)
                if num_tries >= args.max_retries // 2:
                    sys.exit(1)
                    
            num_tries += 1

        # unlock(image_folder_path)
        # unlock(pose_folder_path)
        logging.info('Done for {}'.format(image_folder_path))

    # After rendering all images, combine the JSON files for each scene into a
    # single JSON file.
    # all_scenes = []
    # for scene_path in all_scene_paths:
    #     with open(scene_path, 'r') as f:
    #         all_scenes.append(json.load(f))
    # output = {
    #     'info': {
    #         'date': args.date,
    #         'version': args.version,
    #         'split': args.split,
    #         'license': args.license,
    #     },
    #     'scenes': all_scenes
    # }
    # output_scene_path_file = os.path.join(args.output_dir, 'CATER_scenes.json')
    # with open(args.output_scene_path_file, 'w') as f:
    #     json.dump(output, f)


def rand(L):
    return 2.0 * L * (random.random() - 0.5)


def setup_scene(
    args,
    num_objects=5,
    output_index=0,
    output_image_dir='render.png',
    output_scene_path='render_json',
    first_for_viewpoint=True,
    fixed_objects=[],
    dataset_split=''):
    '''
    Args:
        first_for_viewpoint: Indicates whether or not the scene has already been rendered from a previous camera.
        fixed_objects: If first_for_viewpoint is False, supply the scene objects here.
    '''

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'image_index': output_index,
        'image_filename': os.path.basename(output_image_dir),
        'objects': [],
        'directions': {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(size=5)
    plane = bpy.context.object

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)
    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects['Camera']

    plane_normal = plane.data.vertices[0].normal
    # https://b3d.interplanety.org/en/matrix-vector-and-quaternion-multiplication-in-blender-2-8-python-api/
    cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(
                args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(
                args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(
                args.fill_light_jitter)

    # objects = cup_game(scene_struct, num_objects, args, camera)
    if first_for_viewpoint:
        # Perform actual scene setup.
        objects, blender_objects = add_random_objects(
            scene_struct, num_objects, args, camera)

        record = MovementRecord(blender_objects, args.num_frames + _SCENE_EXTRA_FRAMES)
        if not args.static_objects:
            print('Adding random object motions...')
            allow_contain = args.any_containment and \
                (not(args.containment_even_only) or output_index % 2 == 0) and \
                (args.train_containment or dataset_split.lower() != 'train')
            if not allow_contain:
                print('=> allow_contain is False')
            actions.random_objects_movements(
                objects, blender_objects, args, args.num_frames + _SCENE_EXTRA_FRAMES, args.min_dist,
                record, max_motions=args.max_motions, allow_contain=allow_contain,
                snitch_move_prob=0.993 if args.vagabond_snitch else None)
        movements = record.get_dict()
    
    else:
        # For JSON creation with same scene under different camera viewpoints.
        objects = fixed_objects[0]
        blender_objects = fixed_objects[1]
        movements = fixed_objects[2]  # This contains info about containments over frame ranges.

    # Store the scene data structure to a JSON file.
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    scene_struct['movements'] = movements
    with open(output_scene_path, 'w') as f:
        json.dump(scene_struct, f, indent=2)

    return blender_objects


# This produces individual RGB-D frames instead of video files!
def render_scene(
        args,
        num_objects=5,
        output_index=0,
        output_image_dir=None,
        output_scene_path=None,
        output_blend_path=None,
        output_pose_dir=None,
        dataset_split=None):

    # Ensure directories exist.
    mkdir_p(str(pathlib.Path(output_scene_path).parent))
    if args.save_blendfiles:
        mkdir_p(str(pathlib.Path(output_blend_path).parent))

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    bpy.ops.screen.frame_jump(end=False)
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if args.export_png:
        render_args.filepath = output_image_dir + '.png'  # Overwritten later.
        render_args.image_settings.file_format = 'PNG'
    else:
        raise RuntimeError("I haven't looked at non-png export in a while so this probably doesn't work correctly.")
        pass

    # Video parameters.
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = args.num_frames + _SCENE_EXTRA_FRAMES  # same as kinetics
    bpy.context.scene.frame_step = 1
    render_args.fps = args.fps
    
    if args.cpu is False:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'
            # # In case more than 1 device passed in, use only the first one
            # Not effective, CUDA_VISIBLE_DEVICES before running singularity
            # works fastest.
            # if len(cycles_prefs.devices) > 2:
            #     for device in cycles_prefs.devices:
            #         device.use = False
            #     cycles_prefs.devices[1].use = True
            #     print('Too many GPUs ({}). Using {}. Set only 1 before '
            #           'running singularity.'.format(
            #               len(cycles_prefs.devices),
            #               cycles_prefs.devices[1]))

    # Some CYCLES-specific stuff.
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.cpu is False:
        print('CYCLES is using GPU')
        bpy.context.scene.cycles.device = 'GPU'
    else:
        print('CYCLES is using CPU')

    # Print available hardware for rendering.
    print(bpy.context.preferences.addons['cycles'].preferences.get_devices())

    # Setup scene.
    if output_blend_path is not None and os.path.exists(output_blend_path):
        # Use existing Blender scene file.
        logging.info('Loading pre-defined BLEND file from {}'.format(output_blend_path))
        bpy.ops.wm.open_mainfile(filepath=output_blend_path)
    
    print_camera_matrix()
        
    # Get focal length value and other useful parameters.
    print('Camera count:', len(bpy.data.cameras.values()))
    print('Camera lens:', bpy.data.cameras.values()[0].lens)
    print('Camera lens_unit:', bpy.data.cameras.values()[0].lens_unit)
    print('Camera angle:', bpy.data.cameras.values()[0].angle)
    print('Camera angle_x:', bpy.data.cameras.values()[0].angle_x)
    print('Camera angle_y:', bpy.data.cameras.values()[0].angle_y)

    # Callees take care of scene setup.
    if args.num_cameras > 1:
        # Render from multiple camera viewpoints.
        render_scene_multi_view(
            args, render_args, num_objects, output_index,
            output_blend_path, output_image_dir, output_scene_path,
            output_pose_dir, dataset_split)
        print('Done rendering (MULTI view)')

    else:
        # Only one viewpoint.
        mkdir_p(output_image_dir)
        mkdir_p(output_pose_dir)
        render_scene_single_view(
            args, render_args, num_objects, output_index,
            output_blend_path, output_image_dir, output_scene_path,
            output_pose_dir, dataset_split)
        print('Done rendering (SINGLE view)')


def render_scene_multi_view(
        args, render_args, num_objects, output_index,
        output_blend_path, output_image_dir, output_scene_path,
        output_pose_dir, dataset_split):
    '''
    Example:
    output_blend_path = ...\GREATER_debug\train\GREATER_000012\blend.blend
    output_image_dir = ...\GREATER_debug\train\GREATER_000012\images
    output_scene_path = ...\GREATER_debug\train\GREATER_000012\scene.json
    output_pose_dir = ...\GREATER_debug\train\GREATER_000012\poses
    '''
    # Get view-specific paths.
    view_image_dir = output_image_dir + '_view{}'
    view_scene_path = output_scene_path[:-5] + '_view{}.json'
    view_pose_dir = output_pose_dir + '_view{}'

    # Setup initial scene to set its contents in stone.
    blender_objects = setup_scene(
        args, num_objects, output_index,
        view_image_dir.format(0), view_scene_path.format(0),
        dataset_split=dataset_split)

    # Save Blender scene file.
    if output_blend_path is not None and not os.path.exists(output_blend_path):
        bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)

    # Create Blender compositor nodes for depth map reading.
    if args.export_depth:
        depth_output_file_node = _build_nodes()

    with open(view_scene_path.format(0), 'r') as f:
        scene_struct = json.load(f)
        objects = scene_struct['objects']
        movements = scene_struct['movements']

    last_loc = None
    last_target = None
    last_roll = None
    add_camera_position(0, None, None, None)

    # Start iteration over camera viewpoints.
    last_phis = []
    for view_idx in range(1, args.num_cameras + 1):
        
        # Update scene..
        # Bug related to this call makes instance segmentation impossible..?
        # blender_objects = setup_scene(
        #     args, num_objects, output_index,
        #     view_image_dir.format(view_idx),
        #     view_scene_path.format(view_idx),
        #     first_for_viewpoint=False, fixed_objects=[objects, blender_objects, movements])

        render_all_frames(args, blender_objects, depth_output_file_node,
                          view_image_dir.format(view_idx), view_pose_dir.format(view_idx),
                          view_idx=view_idx, dataset_split=dataset_split, last_phis=last_phis)
        

def render_scene_single_view(
        args, render_args, num_objects, output_index,
        output_blend_path, output_image_dir, output_scene_path,
        output_pose_dir, dataset_split):
    
    # Instantiate scene content once.
    blender_objects = setup_scene(
        args, num_objects, output_index,
        output_image_dir, output_scene_path,
        dataset_split=dataset_split)

    # Save Blender scene file.
    if output_blend_path is not None and not os.path.exists(output_blend_path):
        bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)

    # Create Blender compositor nodes for depth map reading.
    if args.export_depth:
        depth_output_file_node = _build_nodes()

    assert args.export_png, 'export_png must be enabled for single-view export.'
   
   # Start actual render.
    max_num_render_trials = 5
    if args.render:
        while max_num_render_trials > 0:
            try:
                render_all_frames(args, blender_objects, depth_output_file_node,
                                  output_image_dir, output_pose_dir, dataset_split=dataset_split)
                break

            except Exception as e:
                max_num_render_trials -= 1
                print('Render attempt failed')
                print(e)
                traceback.print_exc()
                sys.exit(1)


def render_all_frames(args, blender_objects, depth_output_file_node,
                      output_image_dir, output_pose_dir, view_idx=None,
                      dataset_split=None, last_phis=[]):
    # New code:
    # https://blender.stackexchange.com/questions/112266/how-to-get-camera-matrix-frame-by-frame-in-animation
    # https://github.com/JavonneM/BlenderToRGBD/blob/master/generateRGBD.py
    matrices_P = []
    matrices_K = []
    matrices_RT = []
    snitch_proximities = []  # Euclidean distance to nearest object at every frame.

    if args.random_camera_motion:
        print('Adding random camera motions...')
        add_random_camera_motion(args)
    else:
        print('Setting fixed camera starting position...')
        last_loc, last_target, last_roll, last_phis = get_new_camera_position(args, view_idx, last_phis=last_phis)
        print('last_phis:', last_phis)
        add_camera_position(0, last_loc, last_target, last_roll)
                  
    # Remove any previous temporary EXR files.
    if args.export_depth:
        depth_dp = os.path.join(output_image_dir + '_depth_tmp/')
        depth_output_file_node.base_path = depth_dp
        shutil.rmtree(depth_dp, ignore_errors=True)

    # Render RGB-D.
    bpy.context.scene.frame_end = args.num_frames + _SCENE_EXTRA_FRAMES
    print('Frame range (inclusive):',
          0, args.num_frames + _SCENE_EXTRA_FRAMES, args.speed_factor)
            
    # Render to AVI if specified.
    # if not args.export_png:
    #     file_path = output_image_dir + '.avi'
    #     bpy.context.scene.render.filepath = file_path
    #     bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG'
    #     with suppress_stdout():
    #         bpy.ops.render.render(animation=True, write_still=False)

    # Iterate over all frames for this camera.
    for f in range(0, args.num_frames + 1, args.speed_factor):
        
        bpy.context.scene.frame_set(f)
        camera = bpy.data.objects['Camera']
        P, K, RT = get_3x4_P_matrix_from_blender(camera)
        snitch_prox = _get_snitch_proximity(blender_objects)
        print('Camera matrix (P):', P)
        print('Camera intrinsic (K):', K)
        print('Camera extrinsic (RT):', RT)
        print('Snitch proximity:', snitch_prox)
        matrices_P.append(P)
        matrices_K.append(K)
        matrices_RT.append(RT)
        snitch_proximities.append(snitch_prox)

        # Render to PNG.
        time_idx = f // args.speed_factor
        if args.export_png or f == 0:
            file_path = os.path.join(output_image_dir, str(time_idx).zfill(4) + '.png')
            bpy.context.scene.render.filepath = file_path
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            print('Saving RGB PNG frame to:', file_path)
            with suppress_stdout():
                bpy.ops.render.render(animation=False, write_still=True)

            # Also save depth map.
            if args.export_depth:
                # OLD (numpy file, full precision):
                # NOTE: Must use a custom compiled variant of Blender for this to work, which is quite inconvenient.
                # https://blender.stackexchange.com/questions/69230/python-render-script-different-outcome-when-run-in-background/81240
                # depth_map = _get_depth_map()
                # print('depth_map min, mean, max:', np.min(depth_map),
                #         np.mean(depth_map), np.max(depth_map))
                # file_path = os.path.join(output_image_dir, str(time_idx).zfill(4) + '_depth.npy')
                # np.save(file_path, depth_map)

                # NEW:
                # First part handled by CompositorNodeOutputFile in _build_nodes().
                time.sleep(0.05)
                src_fns = os.listdir(depth_dp)
                assert len(src_fns) == 1
                src_fn = src_fns[0]
                assert str(time_idx * args.speed_factor).zfill(4) in src_fn
                src_fp = os.path.join(depth_dp, src_fn)
                dst_fp = os.path.join(output_image_dir, str(time_idx).zfill(4) + '_depth.exr')
                shutil.move(src_fp, dst_fp)

    # Render instance segmentation by random coloring.
    to_export_flat = args.export_flat and \
        (not(args.flat_val_test_only) or \
            dataset_split is None or len(dataset_split) == 0 or \
            'val' in dataset_split.lower() or 'test' in dataset_split.lower())
    if to_export_flat:
        _render_instance_segmentation(args, bpy.context.scene, blender_objects, output_image_dir)

    matrices_P = np.stack(matrices_P)
    matrices_K = np.stack(matrices_K)
    matrices_RT = np.stack(matrices_RT)
    snitch_proximities = np.array(snitch_proximities)
    print('Saving camera matrices:', matrices_P.shape)
    np.save(os.path.join(output_pose_dir, 'camera_P.npy'), matrices_P)
    print('Saving camera intrinsics:', matrices_K.shape)
    np.save(os.path.join(output_pose_dir, 'camera_K.npy'), matrices_K)
    print('Saving camera extrinsics:', matrices_RT.shape)
    np.save(os.path.join(output_pose_dir, 'camera_RT.npy'), matrices_RT)
    print('Saving snitch proximities:', snitch_proximities.shape)
    np.savetxt(os.path.join(output_pose_dir, 'snitch_proximities.txt'), snitch_proximities)

    # Remove any newly generated temporary EXR files.
    if args.export_depth:
        shutil.rmtree(depth_dp, ignore_errors=True)

    return last_phis


def _build_nodes():
    print('Building nodes...')

    # Switch on nodes.
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # Clear default nodes.
    print('Nodes before:')
    for n in tree.nodes:
        print(n)
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layer = tree.nodes.new('CompositorNodeRLayers')

    # # Create output viewer node (OLD -- requires recompilation).
    # viewer = tree.nodes.new('CompositorNodeViewer')
    # viewer.use_alpha = False
    # # Link Z to output (viewer image).
    # tree.links.new(render_layer.outputs['Depth'], viewer.inputs[0])

    # Create output file node (NEW).
    # https://github.com/cheind/pytorch-blender/blob/develop/pkg_blender/blendtorch/btb/renderer.py
    # Will be stored in output_dir/depth_tmp/0123.exr.
    depth_dp = os.path.join(args.output_dir, 'depth_tmp/')  # Overwritten later.
    output_file = tree.nodes.new("CompositorNodeOutputFile")
    output_file.base_path = depth_dp
    output_file.format.file_format = 'OPEN_EXR_MULTILAYER'
    output_file.format.exr_codec = 'NONE'
    output_file.format.color_depth = '16'
    # Link Z to output (file).
    tree.links.new(render_layer.outputs['Depth'], output_file.inputs['Image'])

    print('Nodes after:')
    for n in tree.nodes:
        print(n)

    return output_file


def _get_depth_map():
    pixels = bpy.data.images['Viewer Node'].pixels
    pixels = np.array(pixels[:])
    print('pixels shape:', pixels.shape)

    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    pixels = pixels.reshape((height, width, 4))

    # Reduce grayscale RGBA to one dimension.
    pixels = pixels[:, :, 0]

    return pixels


def _render_instance_segmentation(args, scn, blender_objects, output_image_dir):
    # https://github.com/kexinyi/ns-vqa/issues/4
    # https://github.com/facebookresearch/clevr-dataset-gen/blob/master/image_generation/render_images.py

    # Cache the render args we are about to modify.
    render_args = bpy.context.scene.render
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_num_samples = bpy.context.scene.cycles.samples
    # old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading.
    # ('BLENDER_EEVEE', 'BLENDER_WORKBENCH', 'CYCLES')
    # render_args.engine = 'BLENDER_RENDER'
    # render_args.engine = 'BLENDER_EEVEE'
    # render_args.engine = 'BLENDER_WORKBENCH'
    render_args.engine = 'CYCLES'
    # render_args.use_antialiasing = False
    bpy.context.scene.cycles.samples = 2  # Doesn't need to be high.
    
    # Move the lights and ground to layer 2 so they don't render.
    # ['Area', 'Axes', 'Camera', 'Cone_0', 'Cone_1', 'Cone_2', 'Cone_3', 'Empty', 'Ground', 'Lamp_Back', 'Lamp_Fill', 'Lamp_Key', 'SmoothCube_v2_0', 'SmoothCylinder-Short_0', 'Sphere_0', 'Sphere_1', 'Sphere_2', 'Spl_0']
    # NOTE: blender_objects contains actual objects of interest only, no environmental stuff.
    bpy.data.objects['Axes'].hide_render = True
    bpy.data.objects['Ground'].hide_render = True

    # Add random shadeless materials to all objects.
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        try:
            bpy.ops.material.new()
        except:
            print('bpy.ops.material.new() failed')
            continue
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        hue = (i * 0.55) % 1.0  # NOTE: Only works well with <20 objects in the scene.
        rgb = tuple(colorsys.hsv_to_rgb(hue, 1.0, 1.0))
        print(rgb)
        object_colors.add(rgb)
        mat.diffuse_color = [*rgb, 1.0]
        mat.specular_color = [*rgb]
        
        # https://blender.stackexchange.com/questions/131015/where-is-shadeless-material-option-for-blender-2-8
        # mat.use_shadeless = True
        # mat.shadow_method = 'NONE'
        # bpy.context.object.active_material.shadow_method = 'NONE'
        
        # https://blender.stackexchange.com/questions/63288/change-value-of-material-emission-strength-from-python-console-or-script
        # mat.use_nodes = True
        # print(mat.node_tree.nodes.keys())

        # https://blender.stackexchange.com/questions/5668/add-nodes-to-material-with-python
        if 1:
            # Remove default
            mat.node_tree.nodes.remove(mat.node_tree.nodes.get('Principled BSDF'))
            material_output = mat.node_tree.nodes.get('Material Output')
            emission = mat.node_tree.nodes.new('ShaderNodeEmission')
            emission.inputs['Color'].default_value = [*rgb, 1.0]
            emission.inputs['Strength'].default_value = 1.0

            # link emission shader to material
            mat.node_tree.links.new(material_output.inputs[0], emission.outputs[0])

        obj.data.materials[0] = mat

    # object_sets = []
    # if args.scatter_flat:
    #     pass

    # Iterate over all frames for this camera.
    for f in range(0, args.num_frames + 1, args.speed_factor):
        bpy.context.scene.frame_set(f)
        camera = bpy.data.objects['Camera']
        P, K, RT = get_3x4_P_matrix_from_blender(camera)

        # Render to PNG.
        time_idx = f // args.speed_factor
        file_path = os.path.join(output_image_dir, str(time_idx).zfill(4) + '_preflat.png')
        bpy.context.scene.render.filepath = file_path
        print('Saving preflat PNG frame to:', file_path)
        with suppress_stdout():
            bpy.ops.render.render(animation=False, write_still=True)

        # Also render just snitch if desired.
        # NOTE: you can also modify this code to render every object in isolation, not just
        # snitch.
        if args.scatter_flat:
            for obj in blender_objects:
                if not 'spl' in obj.name.lower():
                    obj.hide_render = True
            file_path = os.path.join(output_image_dir, str(time_idx).zfill(4) + '_preflat_snitch.png')
            bpy.context.scene.render.filepath = file_path
            print('Saving preflat snitch-only PNG frame to:', file_path)
            with suppress_stdout():
                bpy.ops.render.render(animation=False, write_still=True)
            for obj in blender_objects:
                obj.hide_render = False
        
    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    bpy.data.objects['Axes'].hide_render = False
    bpy.data.objects['Ground'].hide_render = False

    # Set the render settings back to what they were
    render_args.engine = old_engine
    # render_args.use_antialiasing = old_use_antialiasing
    bpy.context.scene.cycles.samples = old_num_samples


def _get_snitch_proximity(blender_objects):
    snitch_xyz = np.zeros(3)
    for obj in blender_objects:
        if 'spl' in obj.name.lower():
            snitch_xyz = np.array(obj.location)
            break
    result = 1e9
    for obj in blender_objects:
        if not 'spl' in obj.name.lower():
            other_xyz = np.array(obj.location)
            cur_dist = np.sqrt(np.sum(np.square(snitch_xyz - other_xyz)))
            result = min(cur_dist, result)
    return result


def print_camera_matrix():
    np.set_printoptions(precision=2, suppress=True)
    # From:
    # https://blender.stackexchange.com/questions/16472/how-can-i-get-the-cameras-projection-matrix
    camera = bpy.data.objects['Camera']
    render = bpy.context.scene.render
    modelview_matrix = camera.matrix_world.inverted()
    projection_matrix = camera.calc_matrix_camera(
        bpy.data.scenes["Scene"].view_layers["RenderLayer"].depsgraph,
        x=render.resolution_x,
        y=render.resolution_y,
        scale_x=render.pixel_aspect_x,
        scale_y=render.pixel_aspect_y,
    )
    final_mat = projection_matrix @ modelview_matrix
    print('Projection matrix:', projection_matrix)
    print('Model view matrix:', modelview_matrix)
    print('Overall camera matrix:', final_mat)


# Build intrinsic camera parameters from Blender camera data
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    print('f_in_mm:', f_in_mm)
    print('s_u:', s_u)
    print('s_v:', s_v)
    print('resolution_x_in_px:', resolution_x_in_px)
    print('resolution_y_in_px:', resolution_y_in_px)
    print('scale:', scale)
    print('sensor_width_in_mm:', sensor_width_in_mm)
    print('sensor_height_in_mm:', sensor_height_in_mm)
    print('pixel_aspect_ratio:', pixel_aspect_ratio)

    K = Matrix(
        ((alpha_u, skew, u_0),
         (0, alpha_v, v_0),
         (0, 0, 1)))
    return K


# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    '''
    Returns camera rotation and translation matrices from Blender.
    The returned matrix describes the center of the world with respect to
    the camera's own coordinate system.
    '''
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
    ))
    return RT


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K @ RT, K, RT


# ----------------------------------------------------------
# Alternative 3D coordinates to 2D pixel coordinate projection code.
# Adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# To have the y axes pointing up and origin at the top-left corner.
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))


def sufficiently_different(phi, last_phis, margin=0.785):
    # Mar 2021: margin = pi/6 radians.
    # Apr 2021: margin = pi/4 radians.
    for other in last_phis:
        if np.abs(phi - other) < margin or np.abs(phi - other - np.pi * 2.0) < margin or np.abs(phi - other + np.pi * 2.0) < margin:
            return False
    return True


def get_new_camera_position(args, view_idx, last_phis=[]):
    '''
    Selects a new random camera (x, y, z) around the scene,
    or picks a fixed position according to view index.
    NOTE: Does not actually modify any Blender state.

    Args:
        view_idx: 1-based camera viewpoint index.
    Returns:
        (source point [meters]), (target point [meters]), roll angle [radians], [last_phis] [radians].
    '''

    if not(args.random_camera_motion) and args.front_back_bird:
        # Impose predetermined view sequence.
        # Front, back, bird, right, left.
        fixed_view_sequence = [(12, 0, 8), (-12, 0, 8), (0, 0, 16), (0, 12, 8), (0, -12, 8)]  # Exported dataset Jan 2021.
        # fixed_view_sequence = [(13, 0, 9), (-13, 0, 9), (0, 0, 17), (0, 13, 9), (0, -13, 9)]  # Proposal?
        new_x, new_y, new_z = fixed_view_sequence[view_idx - 1]
        new_target_x = 0.0
        new_target_y = 0.0
        new_target_z = 1.0
        new_roll = 0.0

    elif not(args.random_camera_motion) and args.random_static_cameras:
        # Used from March 2021 onwards.
        # Find horizontal angle that is far enough from all existing viewpoints in this scene.
        phi_ok = False
        num_tries = 0
        while not phi_ok and num_tries <= 10:
            phi = np.random.rand() * 2.0 * np.pi  # In range [0, 2pi] rad.
            phi_ok = sufficiently_different(phi, last_phis)
            num_tries += 1
        
        # Mar 2021:
        # theta = np.pi / 24.0 + np.random.rand() * np.pi * 3.0 / 24.0  # In range [pi/24, pi/6] rad.
        # Apr 2021:
        theta = np.pi / 18.0 + np.random.rand() * np.pi * 2.0 / 18.0  # In range [pi/18, pi/6] rad.
        
        new_x = np.cos(phi) * np.cos(theta) * args.camera_radius
        new_y = np.sin(phi) * np.cos(theta) * args.camera_radius
        new_z = np.sin(theta) * args.camera_radius
        new_target_x = 0.0
        new_target_y = 0.0
        new_target_z = 1.5
        new_roll = 0.0
        last_phis.append(phi)

    else:
        # Randomly select next camera location.
        # Don't move in X and Y at the same time, as it could cross the (0, 0, z) point,
        # which is a singularity.
        new_x, new_y, new_z = None, None, None
        if np.random.random() > 0.5:
            # Move in X.
            new_x = np.random.choice([-10, 10])
        else:
            # Move in Y.
            new_y = np.random.choice([-10, 10])
        
        # Move in Z.
        if args.extra_vertical_movement <= 0.0:
            new_z = np.random.choice([8, 10, 12])
            new_target_z = 0.0
        else:
            offset = np.random.random()
            new_z = 12.0 - offset * (args.extra_vertical_movement + 4.0)
            offset = (offset + np.random.random()) / 2.0
            new_target_z = offset * args.extra_vertical_movement / 2.0
        
        # Set target and tilt.
        new_target_x = (np.random.random() * 2.0 - 1.0) * args.random_horizontal_targets
        new_target_y = (np.random.random() * 2.0 - 1.0) * args.random_horizontal_targets
        new_roll = np.random.random() * args.random_roll_degrees * np.pi / 180.0
        
        # Ensure scene remains nearly fully visible.
        origin_scale = 1.0 + args.extra_vertical_movement / 32.0 + args.random_horizontal_targets / 32.0
        if new_x is not None:
            new_x *= origin_scale
        if new_y is not None:
            new_y *= origin_scale
    
    return (new_x, new_y, new_z), (new_target_x, new_target_y, new_target_z), new_roll, last_phis


def add_random_camera_motion(args):

    # Now go through these locations in a random order.
    shift_interval = 30
    
    # Start from the same position everytime, as I want to be able to track positions.
    last_loc = None
    last_target = None
    last_roll = None
    add_camera_position(0, None, None, None)  # Doesn't overwrite location => scene default?
    
    # Periodically set random position.
    for frame_idx in range(shift_interval, args.num_frames + _SCENE_EXTRA_FRAMES + shift_interval - 1, shift_interval):
        last_loc, last_target, last_roll, _ = get_new_camera_position(args, None)
        add_camera_position(frame_idx, last_loc, last_target, last_roll)
    
    # Ensure last frame is set correctly.
    if last_loc is not None:
        add_camera_position(args.num_frames + _SCENE_EXTRA_FRAMES, last_loc, last_target, last_roll)


def add_camera_position(frame_idx, loc, target, roll):
    '''
    Inserts a keyframe at the chosen time to ensure the camera position is set to the desired parameters.
    '''
    print('Call add_camera_position()', frame_idx, loc, target, roll)
    time.sleep(0.20)
    
    obj = bpy.data.objects['Camera']
    if loc is not None and loc[0] is not None:
        obj.location.x = loc[0]
    if loc is not None and loc[1] is not None:
        obj.location.y = loc[1]
    if loc is not None and loc[2] is not None:
        obj.location.z = loc[2]
    if target is not None and target[0] is not None:
        camera_look_at(obj, target, frame_idx)
    if roll is not None:
        # TODO
        pass
    
    # Ensure no conflicting keyframes exist.
    try:
        obj.keyframe_delete(data_path='location', frame=frame_idx)
    except RuntimeError:
        print('keyframe_delete() failed')
        time.sleep(0.20)

    # Now insert keyframe.
    obj.keyframe_insert(data_path='location', frame=frame_idx)
    # print_camera_matrix()


def camera_look_at(obj_camera, target, frame_idx):
    # The Blender scene file reveals that the camera has a 'Track To' object constraint.
    # https://blender.stackexchange.com/questions/132825/python-selecting-object-by-name-in-2-8
    bpy.data.objects['Empty'].location.x = target[0]
    bpy.data.objects['Empty'].location.y = target[1]
    bpy.data.objects['Empty'].location.z = target[2]
    bpy.data.objects['Empty'].keyframe_insert(data_path='location', frame=frame_idx)

    # https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
    # loc_camera = obj_camera.matrix_world.to_translation()
    # target = Vector(target)
    # direction = target - loc_camera
    # # point the cameras '-Z' and use its 'Y' as up
    # rot_quat = direction.to_track_quat('-Z', 'Y')
    # # assume we're using euler rotation
    # obj_camera.rotation_euler = rot_quat.to_euler()


def add_random_objects(scene_struct, num_objects, args, camera, recursive_calls=0):
    """
    Add random objects to the current blender scene
    """

    if recursive_calls >= args.max_retries // 2:
        raise RuntimeError('This scene is impossible')

    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = list(properties['sizes'].items())
        print('size_mapping:', size_mapping)

    shape_color_combos = None
    if args.shape_color_combos_json is not None:
        with open(args.shape_color_combos_json, 'r') as f:
            shape_color_combos = list(json.load(f).items())

    positions = []
    objects = []
    blender_objects = []
    for i in range(num_objects):
        
        # Follow CATER tradition if not args.fully_random_objects.
        if i == 0 and not args.fully_random_objects:
            # first element is the small shiny gold "snitch"!
            # size_name, r = "small", 0.3  # slightly larger than small
            size_name, r = "small", 0.285  # reduced by 5% for vbsnitch_3
            obj_name, obj_name_out = 'Spl', 'spl'
            color_name = "gold"
            rgba = [1.0, 0.843, 0.0, 1.0]
            mat_name, mat_name_out = "MyMetal", "metal"
        elif i == 1 and not args.fully_random_objects:
            # second element is a medium cone
            size_name, r = "medium", 0.5
            obj_name, obj_name_out = 'Cone', 'cone'
            color_name, rgba = random.choice(list(
                color_name_to_rgba.items()))
            mat_name, mat_name_out = random.choice(material_mapping)
        elif i == 2 and not args.fully_random_objects:
            # third element is a large cone
            size_name, r = "large", 0.75
            obj_name, obj_name_out = 'Cone', 'cone'
            color_name, rgba = random.choice(list(
                color_name_to_rgba.items()))
            mat_name, mat_name_out = random.choice(material_mapping)
        
        else:
            # Create totally random objects instead if args.fully_random_objects.
            # Choose a random size
            size_name, r = random.choice(size_mapping)
            # Choose random color and shape
            if shape_color_combos is None:
                obj_name, obj_name_out = random.choice(object_mapping)
                color_name, rgba = random.choice(list(
                    color_name_to_rgba.items()))
            else:
                obj_name_out, color_choices = random.choice(shape_color_combos)
                color_name = random.choice(color_choices)
                obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
            rgba = color_name_to_rgba[color_name]
            # Choose a random material
            mat_name, mat_name_out = random.choice(material_mapping)

        # Size modification.
        # NOTE: Large values might make adhering to min_dist and margin more difficult!
        r *= args.size_multiplier

        # Try to place the object, ensuring that we don't intersect any
        # existing objects and that we are more than the desired margin away
        # from all existing objects along all cardinal directions.
        num_tries = 0
        while num_tries <= args.max_retries + 1:
            
            # If we try and fail to place an object too many times, then
            # delete all the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                print('=> max_retries exceeded => starting over...')
                time.sleep(0.10)
                return add_random_objects(scene_struct, num_objects, args,
                                          camera, recursive_calls=recursive_calls+1)
            
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            # Check to make sure the new object is further than min_dist from
            # all other objects, and further than margin along the four
            # cardinal directions
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < args.min_dist:
                    dists_good = False
                    break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = scene_struct['directions'][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin:
                        logging.debug('{} {} {}'.format(
                            margin, args.margin, direction_name))
                        logging.debug('BROKEN MARGIN!')
                        margins_good = False
                        break
                if not margins_good:
                    break
            if dists_good and margins_good:
                break

        # For cube, adjust the size a bit
        if obj_name == 'Cube':
            r /= math.sqrt(2)

        # Choose random orientation for the object.
        theta = 360.0 * random.random()

        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # Actually add material
        utils.add_material(mat_name, Color=rgba)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape': obj_name_out,
            'size': size_name,
            'sized': r,
            'material': mat_name_out,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
            'instance': obj.name,
        })
    return objects, blender_objects


def cup_game(scene_struct, num_objects, args, camera):
    # make some random objects
    # objects, blender_objects = add_random_objects(
    objects, blender_objects = add_cups(
        scene_struct, num_objects, args, camera)
    bpy.ops.screen.frame_jump(end=False)
    # from https://blender.stackexchange.com/a/70478
    add_flips(blender_objects, num_flips=args.num_flips,
              total_frames=args.num_frames + _SCENE_EXTRA_FRAMES)
    animate_camera(args.num_frames + _SCENE_EXTRA_FRAMES)
    return objects


def animate_camera(num_frames):
    path = [
        (0, -10, 10),
        (-10, 0, 10),
        (0, 10, 10),
        (10, 0, 5),
    ]
    shift_interval = 20
    cur_pos_id = -1
    obj = bpy.data.objects['Camera']
    for frame_idx in range(0, num_frames + shift_interval - 1, shift_interval):
        obj.keyframe_insert(data_path='location', frame=frame_idx)
        cur_pos_id = (cur_pos_id + 1) % len(path)
        obj.location.x = path[cur_pos_id][0]
        obj.location.y = path[cur_pos_id][1]
        obj.location.z = path[cur_pos_id][2]


def add_cups(scene_struct, num_objects, args, camera):
    """
    Add random objects to the current blender scene
    """
    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = list(properties['sizes'].items())

    # shape_color_combos = None
    # if args.shape_color_combos_json is not None:
    #     with open(args.shape_color_combos_json, 'r') as f:
    #         shape_color_combos = list(json.load(f).items())

    positions = []
    objects = []
    blender_objects = []

    # Choose a random size, same for all cups
    size_name, r = size_mapping[0]
    first_cup_x = 0
    first_cup_y = 0

    # obj_name, obj_name_out = random.choice(object_mapping)
    obj_name, obj_name_out = [el for el in object_mapping
                              if el[1] == 'cylinder'][0]
    color_name, rgba = random.choice(list(color_name_to_rgba.items()))

    # If using combos
    # obj_name_out, color_choices = random.choice(shape_color_combos)
    # color_name = random.choice(color_choices)
    # obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
    # rgba = color_name_to_rgba[color_name]

    # Attach a random material
    mat_name, mat_name_out = random.choice(material_mapping)

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
        r /= math.sqrt(2)

    # Choose random orientation for the object.
    # theta = 360.0 * random.random()
    theta = 0.0

    for i in range(num_objects):
        x = first_cup_x + i * 1.5
        y = first_cup_y

        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        utils.add_material(mat_name, Color=rgba)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape': obj_name_out,
            'size': size_name,
            'material': mat_name_out,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
        })
    return objects, blender_objects


def add_flips(blender_objects, num_flips=10, total_frames=300):
    # add current locations as a keyframe
    current_frame = 0
    bpy.context.scene.frame_set(current_frame)
    for obj in blender_objects:
        obj.keyframe_insert(data_path='location')

    frames_per_flip = total_frames // num_flips
    for flip_id in range(num_flips):
        # select random 2 cups to flip
        end_frame = min(current_frame + frames_per_flip - 1, total_frames)
        actions.add_flip(
            blender_objects, start_frame=current_frame, end_frame=end_frame)
        current_frame = end_frame + 1
    bpy.ops.screen.frame_jump(end=False)


def compute_all_relationships(scene_struct, eps=0.2):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i]
    then object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below':
            continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2:
                    continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
                all_relationships[name].append(sorted(list(related)))
    return all_relationships


if __name__ == '__main__':

    if _INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()


        if not _RUNNING_ON_SERVER:
            args = parser.parse_args(argv)

        else:
            args = parser.parse_args()



        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        main(args)

    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:

        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')
