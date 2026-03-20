from picamera2 import Picamera2
from time import sleep
from datetime import datetime
import os

picam2 = Picamera2()

camera_config = picam2.create_still_configuration({"size" : (3200, 1800)})
picam2.configure(camera_config)
picam2.start()
sleep(2)
n = 80

folder_name = datetime.now().strftime('%Y-%m-%d')
folder_name = "../DATA/" + folder_name
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

for x in range(n):
    time = datetime.now().strftime('%Y-%m-%d_%H%M')
    picam2.capture_file(folder_name+"/" + time + ".jpg")
    print("Sucessfully captured image in:", time)
    print(f'{x+1} of {n}')
    sleep(600)
