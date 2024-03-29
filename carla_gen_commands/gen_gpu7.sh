# Auto-generated by dataset_cmds.py

DSETROOT="carla_4d/"
PORT=6172

for REDO_IDX in {1..5}
do
echo $REDO_IDX


SPLIT="train"
NAME="train_0007"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 7 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0015"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 15 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0023"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 0 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0031"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 8 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0039"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 16 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0047"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 1 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0055"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 9 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0063"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 17 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0071"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 2 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0079"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 10 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0087"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 18 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0095"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 3 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0103"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 11 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0111"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 19 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0119"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 4 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0127"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 12 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0135"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 20 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0143"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0151"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 13 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0159"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 21 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0167"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0175"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 14 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0183"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0191"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 7 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0199"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 15 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0207"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 0 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0215"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 8 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0223"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 16 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0231"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 1 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0239"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 9 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0247"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 17 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0255"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 2 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0263"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 10 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0271"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 18 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0279"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 3 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0287"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 11 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0295"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 19 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0303"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 4 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0311"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 12 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0319"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 20 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0327"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0335"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 13 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0343"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 21 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0351"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0359"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 14 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0367"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0375"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 7 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0383"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 15 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0391"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 0 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0399"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 8 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0407"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 16 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0415"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 1 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0423"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 9 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0431"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 17 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0439"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 2 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0447"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 10 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0455"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 18 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0463"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 3 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0471"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 11 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0479"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 19 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0487"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 4 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0495"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town10HD --num_vehicles 77 --num_walkers 100 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 12 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


done

