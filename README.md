# AutonomousBoat

Vision-based autonomous RC boat for swimming pool navigation. Uses a Raspberry Pi 4 with a CSI camera and an I2C IMU to detect obstacles (walls, floating objects, people) and steer around them with differential thrust.

## Hardware

- **Raspberry Pi 4 Model B** running Raspberry Pi OS Trixie (64-bit)
- **Pi Camera v1** (OmniVision OV5647, 5MP, CSI ribbon)
- **LSM6DSO IMU** (6-axis accel + gyro, I2C address `0x6A`)
- Hull: AliExpress 2.4G twin-motor RC racing boat, stock electronics replaced
- Motor driver: TBD (DRV8833 or TB6612FNG)
- Buck converter: TBD (MP1584EN, 5.1V output)
- Battery: TBD (2S 7.4V LiPo)

## Software setup

Assumes a fresh Pi OS Trixie install.

### System packages

```bash
sudo apt install -y i2c-tools python3-dev python3-picamera2 python3-rpi-lgpio
sudo raspi-config nonint do_i2c 0
echo "i2c-dev" | sudo tee -a /etc/modules
```

### Python environment

```bash
cd ~
python3 -m venv --system-site-packages autoboat-env
source autoboat-env/bin/activate
pip install \
    adafruit-circuitpython-lsm6ds \
    adafruit-circuitpython-busdevice \
    adafruit-circuitpython-register \
    adafruit-platformdetect \
    Adafruit-PureIO \
    opencv-python-headless \
    Pillow
pip install --no-deps adafruit-blinka
```

`--system-site-packages` lets the venv see the apt-installed `picamera2`.
`--no-deps` on `adafruit-blinka` skips `rpi_ws281x` and `RPi.GPIO` which conflict with the modern `rpi-lgpio` on Trixie. The IMU only uses I2C so neither is needed.

### Verify

```bash
source ~/autoboat-env/bin/activate
python3 -c "from picamera2 import Picamera2; from adafruit_lsm6ds.lsm6dsox import LSM6DSOX; import board, busio; print('ok')"
```

## Project structure

```
autoboat/
├── sensors/
│   ├── imu.py              # Threaded LSM6DSO reader, ~100 Hz
│   └── camera.py           # Threaded Picamera2 wrapper, ~30 fps
├── vision/
│   └── pipeline.py         # HSV water segmentation, per-column depth, zone analysis
├── control/                # (not yet implemented)
├── scripts/
│   ├── sensor_rates.py     # Measure achieved IMU + camera rates
│   ├── capture_pool_images.py  # Capture labeled pool test images
│   └── test_pipeline.py    # Run vision pipeline on saved images
├── pool_images/            # Test images of the pool in various conditions
└── logs/                   # Runtime logs (gitignored)
```

## Running the scripts

All scripts assume the venv is active and you're running as user `ben` (in the `i2c`, `gpio`, `video` groups).

**Measure sensor rates:**
```bash
python3 ~/autoboat/scripts/sensor_rates.py
```
Runs both sensors for 5 seconds and reports actual achieved Hz/fps.

**Capture pool test images:**
```bash
python3 ~/autoboat/scripts/capture_pool_images.py
```
Interactive. Prompts for a label, captures a frame, writes a timestamped JPEG to `pool_images/`.

**Run the vision pipeline on saved images:**
```bash
python3 ~/autoboat/scripts/test_pipeline.py
```
Analyzes every JPEG in `pool_images/` and writes annotated output to `pool_images/analyzed/`.

## Vision pipeline output

The pipeline returns a `NavResult` for each frame:

- `mask`: binary water mask
- `depths`: per-column free-water depth in pixels (length = frame width)
- `zones`: 5 horizontal zones with median water depth, left to right
- `best_zone`: zone index (0-4) with the most open water
- `center_depth_pct`: free water ahead in the center column, as % of frame height

Steering logic (when implemented): turn toward `best_zone`, reduce speed when `center_depth_pct` drops below a threshold.

## Sensor architecture

Both sensors run in their own thread with a "latest sample wins" pattern. Consumers call `latest()` and get the most recent reading. No queues, no locks.

- IMU: ~100 Hz, sample age < 10 ms
- Camera: ~32 fps at 320x240, frame age < 32 ms

Verified via `scripts/sensor_rates.py`.

## Status

Working:
- Camera detected and streaming via Picamera2
- IMU detected, reading clean accel + gyro
- Threaded sensor layer hits target rates
- Vision pipeline correctly identifies water vs obstacles in cloudy outdoor light

Not yet implemented:
- Motor driver hardware and software
- Control loop
- Live demo (pipeline against the threaded camera in real time)
- Sunny-condition HSV recalibration
- Temporal smoothing on steering output
- Distance calibration (pixel rows -> meters)
- Hardware watchdog for motor cutoff on control hang
- Waterproof enclosure

## Calibration notes

HSV thresholds in `vision/pipeline.py` are tuned for the OV5647 under cloudy evening light. The camera's auto white balance produces an olive-green color cast for clear pool water (visible in `pool_images/`). Thresholds are matched to what the camera actually outputs, not what the water looks like to the eye.

When lighting changes significantly (midday sun, indoor), recalibration via pixel sampling from new images is needed. Use `analyze_water.py`-style sampling to get new percentile ranges.

## License

Personal project. No license specified.
