# Setup development environment
### 1. Setup:
1. Open the terminal
2. Go to the repository folder "dev"
3. Execute ```docker compose up -d```
4. This should create the following containers:
    - home assistant
    - ollama
    - mongodb
5. Go to "http://localhost:8123" and setup your user

### 2. Install HACS:
1. Open the shell of the home assitant container
2. Execute ```wget -O - https://get.hacs.xyz | bash -``` 
3. Restart the home assistant container
4. Go to the "Settings/Devices & services" and click on "Add integration"
5. Select "HACS"

### 3. Create dummy entities:
1. Go to "HACS" in the sidebar
2. Search for "Virtual Components" and install it
3. Restart the home assistant container
4. Copy the file "dev/devices/virtual.yaml" into "dev/config/"
5. Go to the "Settings/Devices & services" and click on "Add integration"
6. Select "Virtual Components" and click "Submit" and then "Skip and finish"
7. Go to the "dev/devices" folder and execute ```pip3 install -r requirements.txt``` and ```python3 assign_rooms.py``` ([creating long lived access tokens](https://community.home-assistant.io/t/how-to-get-long-lived-access-token/162159/4))
