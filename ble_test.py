import asyncio
from bleak import BleakClient, BleakScanner

# Replace with the BLE address of your ESP32
ESP32_ADDRESS = "A0:A3:B3:2A:D8:22"  # The MAC address of your ESP32
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# Function to send data (1 to turn ON, 0 to turn OFF)
async def send_lamp_command(command):
    print(f"Scanning for devices...")
    
    # Connect to the ESP32
    async with BleakClient(ESP32_ADDRESS) as client:
        print(f"Connected to {ESP32_ADDRESS}")
        
        while True:
            # Send the command to the characteristic
            await client.write_gatt_char(CHARACTERISTIC_UUID, command.encode())
            print(f"Sent command: {command}")
            asyncio.sleep(5)

        await client.disconnect()

# Run the asyncio event loop
if __name__ == "__main__":
    # command = input("Enter '1' to turn lamp ON or '0' to turn lamp OFF: ")
    asyncio.run(send_lamp_command('hello'))
