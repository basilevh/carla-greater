# Auto-generated by dataset_cmds.py

DSETROOT="carla_4d/"
PORT=6102

for REDO_IDX in {1..5}
do
echo $REDO_IDX


SPLIT="train"
NAME="train_0000"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 0 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0008"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 8 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0016"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 16 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0024"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 1 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0032"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 9 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0040"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 17 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0048"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 2 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0056"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 10 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0064"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 18 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0072"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 3 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0080"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 11 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0088"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 19 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0096"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 4 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0104"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 12 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0112"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 20 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0120"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 5 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0128"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 13 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0136"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 21 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0144"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0152"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 14 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0160"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0168"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 7 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0176"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 15 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0184"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 0 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0192"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 8 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0200"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 16 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0208"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 1 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0216"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 9 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0224"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 17 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0232"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 2 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0240"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 10 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0248"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 18 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0256"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 3 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0264"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 11 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0272"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 19 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0280"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 4 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0288"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 12 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0296"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 20 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0304"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 5 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0312"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 13 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0320"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 21 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0328"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0336"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 14 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0344"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0352"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 7 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0360"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 15 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0368"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 0 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0376"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 8 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0384"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 16 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="train"
NAME="train_0392"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 1 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0400"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 9 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0408"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 17 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0416"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 2 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0424"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 10 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0432"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 18 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0440"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 3 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0448"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 11 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0456"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 19 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0464"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 4 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0472"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 12 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0480"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 20 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0488"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 5 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


SPLIT="val"
NAME="val_0496"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town01 --num_vehicles 127 --num_walkers 165 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 13 --multiview 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000 --multiview 5


done

