# Auto-generated by dataset_cmds.py

DSETROOT="carla_4d/"
PORT=6142

for REDO_IDX in {1..5}
do
echo $REDO_IDX


SPLIT="train"
NAME="train_0004"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 4 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0012"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 12 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0020"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 20 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0028"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0036"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 13 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0044"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 21 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0052"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0060"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 14 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0068"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0076"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 7 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0084"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 15 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0092"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 0 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0100"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 8 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0108"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 16 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0116"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 1 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0124"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 9 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0132"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 17 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0140"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 2 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0148"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 10 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0156"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 18 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0164"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 3 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0172"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 11 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0180"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 19 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0188"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 4 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0196"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 12 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0204"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 20 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0212"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0220"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 13 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0228"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 21 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0236"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0244"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 14 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0252"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0260"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 7 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0268"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 15 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0276"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 0 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0284"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 8 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0292"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 16 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0300"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 1 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0308"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 9 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0316"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 17 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0324"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 2 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0332"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 10 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0340"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 18 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0348"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 3 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0356"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 11 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0364"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 19 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0372"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 4 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0380"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 12 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0388"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 20 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="train"
NAME="train_0396"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 5 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0404"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 13 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0412"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 21 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0420"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0428"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 14 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0436"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 6 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0444"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 7 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0452"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 15 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0460"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 0 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0468"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 8 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0476"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 16 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0484"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 1 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


SPLIT="val"
NAME="val_0492"
python record_simulation.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --world Town05 --num_vehicles 151 --num_walkers 196 --safe 1 --num_frames 1000 --fps 10
python capture_sensor_data.py --record_path $DSETROOT/$SPLIT/$NAME/$NAME.log --port $PORT --weather_preset 9 --num_frames 1000 --fps 10 --time_factor 1.0
python video_scene.py --capture_path $DSETROOT/$SPLIT/$NAME --num_frames 1000


done

