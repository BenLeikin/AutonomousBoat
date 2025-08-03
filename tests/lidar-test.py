#!/usr/bin/env python3
import serial, time

def find_frame(ser):
    # look for 0x59 0x59 header
    while True:
        b1 = ser.read(1)
        if not b1 or b1[0] != 0x59:
            continue
        b2 = ser.read(1)
        if b2 and b2[0] == 0x59:
            return ser.read(7)  # rest of frame

def parse_frame(raw):
    if len(raw) != 7:
        return None
    data = bytes([0x59,0x59]) + raw
    checksum = sum(data[0:8]) & 0xFF
    if checksum != data[8]:
        return None
    dist    = data[2]  | (data[3]  << 8)
    strength= data[4]  | (data[5]  << 8)
    temp    = (data[6] | (data[7] << 8)) / 8.0 - 256
    return dist, strength, temp

def main():
    ser = serial.Serial('/dev/ttyS0', 115200, timeout=1)
    time.sleep(2)
    print("Reading Lidar frames (Ctrl+C to exit)")
    try:
        while True:
            tail = find_frame(ser)
            result = parse_frame(tail)
            if result:
                d, s, t = result
                print(f"Distance: {d/100:.2f} m | Strength: {s} | Temp: {t:.1f} °C")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

if __name__ == '__main__':
    main()
