import asyncio
from bleak import BleakScanner
from bleak import BleakClient
from threading import Thread

def main():
    target_name = "ESP32BLE"
    target_address = "78:21:84:7E:30:DE"

    SERVICE_UUID=        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
    CHARACTERISTIC_UUID= "beb5483e-36e1-4688-b7f5-ea07361b26a8"

    # devices = BleakScanner.discover()
    # for d in devices:
    #     print(d)
    #     if target_name == d.name:
    #         target_address = d.address
    #         print("found target {} bluetooth device with address {} ".format(target_name,target_address))
    #         break

    if target_address is not None:        
        with BleakClient(target_address) as client:
            print(f"Connected: {client.is_connected}")
                
            while 1:
                text = 'on'
                if text == "quit":
                    break

                client.write_gatt_char(CHARACTERISTIC_UUID, bytes(text, 'UTF-8'), response=True)
                
                try:
                    data = client.read_gatt_char(CHARACTERISTIC_UUID)
                    data = data.decode('utf-8') #convert byte to str
                    print("data: {}".format(data))
                except Exception:
                    pass
                
            
    else:
        print("could not find target bluetooth device nearby")

ble_thread = Thread(target=main(), args=())
ble_thread.start()