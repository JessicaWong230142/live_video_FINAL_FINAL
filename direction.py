import requests



#Function to send commands to the robot
def send_command(direction):
    ip = '192.168.1.30'
    url = f'http://192.168.1.30:4200/move'
    data = {'direction': direction}

    response = requests.post(url, json=data)
    if response.status_code == 200:
        print(f"Command '{direction}' sent successfully.")
    else:
        print(f"Failed to send command '{direction}'")
