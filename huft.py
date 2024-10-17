import asyncio
from bleak import BleakScanner
from bleak import BleakClient

async def main():
    target_name = "ESP32BLE"
    target_address = "40:91:51:9b:fd:e6"

    SERVICE_UUID=        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
    CHARACTERISTIC_UUID= "beb5483e-36e1-4688-b7f5-ea07361b26a8"

    devices = await BleakScanner.discover()
    for d in devices:
        print(d)
        if target_name == d.name:
            target_address = d.address
            print("found target {} bluetooth device with address {} ".format(target_name,target_address))
            break

    if target_address is not None:        
        async with BleakClient(target_address) as client:
            print(f"Connected: {client.is_connected}")
                
            while 1:
                text = input()
                if text == "quit":
                    break

                await client.write_gatt_char(CHARACTERISTIC_UUID, bytes(text, 'UTF-8'), response=True)
                
                try:
                    data = await client.read_gatt_char(CHARACTERISTIC_UUID)
                    data = data.decode('utf-8') #convert byte to str
                    print("data: {}".format(data))
                except Exception:
                    pass
                
            
    else:
        print("could not find target bluetooth device nearby")


asyncio.run(main())