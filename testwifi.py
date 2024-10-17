import requests

esp32_ip = "http://10.3.51.119/send-data"
data_to_send = "on"

response = requests.post(esp32_ip, data=data_to_send)
if response.status_code == 200:
    print("Data sent successfully:", response.text)
else:
    print("Failed to send data")
