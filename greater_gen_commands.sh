
# In one tmux pane:
python render_videos.py --num_cameras 3 --render_num_samples 32 --num_frames 300 --num_scenes 7000 --train_val_test_split 1 --random_camera_motion 0 --min_objects 8 --max_objects 12 --min_dist 0 --margin 0 --size_multiplier 2.0 --output_dir greater_4d/ --cpu 1 --speed_factor 1 --random_static_cameras 1 --camera_radius 15 --any_containment 0 --vagabond_snitch 1 --start_index 0 --stop_index 100

# In another tmux pane:
python render_videos.py --num_cameras 3 --render_num_samples 32 --num_frames 300 --num_scenes 7000 --train_val_test_split 1 --random_camera_motion 0 --min_objects 8 --max_objects 12 --min_dist 0 --margin 0 --size_multiplier 2.0 --output_dir greater_4d/ --cpu 1 --speed_factor 1 --random_static_cameras 1 --camera_radius 15 --any_containment 0 --vagabond_snitch 1 --start_index 100 --stop_index 200

# And so on...

# Finally, postprocess (you can parallelize this too):
python postprocess_dataset.py --root_dir greater_4d/ --fps 24 --num_processes 16 --write_poses 1 --ignore_if_exist 1 --remove_depth_exr 1 --num_frames 301 --speed_factor 1 --index_from 0 --index_to 99
