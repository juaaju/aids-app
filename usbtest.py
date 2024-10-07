import serial
import time

# Set up the serial connection
ser = serial.Serial('COM6', 115200, timeout=1)

time.sleep(2)  # Wait for the connection to establish

# Send data to ESP32
data_to_send = "Hello ESP32!"
ser.write((data_to_send + '\n').encode('utf-8'))  # Send data

# Optional: Read the response from ESP32
response = ser.readline().decode('utf-8').strip()
print("ESP32 Response:", response)

# Close the serial connection
ser.close()
