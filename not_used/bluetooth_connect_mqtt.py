import paho.mqtt.client as mqtt

# MQTT broker details
mqtt_broker = "broker.hivemq.com"  # Public broker
mqtt_port = 1883
mqtt_topic = "esp/test"

# Define the MQTT client
client = mqtt.Client()

# Connect to the MQTT broker
client.connect(mqtt_broker, mqtt_port, 60)

# Publish a message to the ESP32``
def send_message(message):
    client.publish(mqtt_topic, message)
    print(f"Sent message: {message}")

# Example usage
send_message("Hello from laptop")

# Disconnect from the broker
client.disconnect()
