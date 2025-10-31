"""MQTT Publisher for simulating IIoT machine data."""
import json
import random
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
KEEPALIVE = 60

# Machine configurations
MACHINES = ["M1", "M2", "M3"]
STATES = {
    "RUN": {"weight": 0.7, "power_base": 2.5},
    "IDLE": {"weight": 0.2, "power_base": 0.4},
    "STOP": {"weight": 0.1, "power_base": 0.05},
}


def generate_machine_data(machine_id: str) -> dict:
    """Generate sample machine data with realistic power consumption patterns.
    
    Args:
        machine_id: The ID of the machine
        
    Returns:
        dict: Dictionary containing machine telemetry data
    """
    # Randomly select machine state based on weights
    states, weights = zip(*[(k, v["weight"]) for k, v in STATES.items()])
    status = random.choices(states, weights=weights, k=1)[0]
    
    # Generate power reading with some noise
    base_power = STATES[status]["power_base"]
    power_kw = round(random.normalvariate(base_power, 0.1), 3)
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_id": machine_id,
        "power_kW": power_kw,
        "status": status,
    }


def on_connect(client, userdata, flags, rc):
    """Callback when the client connects to the MQTT broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")


def main():
    """Main function to run the MQTT publisher."""
    # Initialize MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    
    try:
        # Connect to MQTT broker
        client.connect(MQTT_BROKER, MQTT_PORT, KEEPALIVE)
        client.loop_start()
        
        print(f"Publishing machine data for {MACHINES} at 1Hz...")
        print("Press Ctrl+C to stop")
        
        # Main publishing loop
        while True:
            for machine in MACHINES:
                # Generate and publish data for each machine
                data = generate_machine_data(machine)
                topic = f"factory/line1/{machine}/power"
                payload = json.dumps(data)
                
                # Publish with error handling
                try:
                    result = client.publish(topic, payload, qos=0)
                    if result.rc != mqtt.MQTT_ERR_SUCCESS:
                        print(f"Error publishing to {topic}: {result.rc}")
                except Exception as e:
                    print(f"Exception while publishing: {e}")
            
            # Sleep to maintain 1Hz publish rate
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping publisher...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
