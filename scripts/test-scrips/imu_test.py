import time
import board
import busio
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS

i2c = busio.I2C(board.SCL, board.SDA)
sensor = LSM6DS(i2c, address=0x6A)

while True:
    ax, ay, az = sensor.acceleration
    gx, gy, gz = sensor.gyro
    print(f"Accel (m/s^2): {ax:+6.2f} {ay:+6.2f} {az:+6.2f}   "
          f"Gyro (rad/s): {gx:+6.2f} {gy:+6.2f} {gz:+6.2f}")
    time.sleep(0.1)
