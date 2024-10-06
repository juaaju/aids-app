import serial
import time

ser = serial.Serial('COM6', 115200)
time.sleep(5)

def send(command):
    ser.write(command.encode())
i=0
while i<50:
    send('Hello world')
    i+=1
ser.close()