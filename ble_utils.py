import bleak
import asyncio

async def main(characteristic_uuid, command, ble_client=None):
    await ble_client.write_gatt_char(characteristic_uuid, command.encode())
    
async def connect(address):
    async with bleak.BleakClient(address) as client:
        print(f"Connected to {address}")
        return client

async def disconnect(client, address):
    await client.disconnect()
    print(f"Disconnected from {address}")