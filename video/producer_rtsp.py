# from utils.Producer import Producer
from kafka import KafkaProducer
import rtsp
import cv2
import time
import numpy as np
from utils.video_capture import VideoCapture

def run_on_video(video_path, output_video_name, conf_thresh):
    cap = VideoCapture(video_path)
    
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        img_raw = cap.read()
        read_frame_stamp = time.time()
        if (status):
            if idx == 2:
                idx = 0
            else:
                idx += 1
                continue
            encode_param= [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, encode_image = cv2.imencode('.png', img_raw[:, :], encode_param)
            data = np.array(encode_image)
            message = data.tostring()

            producer.send(topic, message)
            # producer.queue.put(message)
            print ('[Send video] Topic: ' + topic)
            
if __name__ == "__main__":
    topic = 'fmd-video-cam1'
    producer = KafkaProducer(bootstrap_servers='192.168.7.130:31090,192.168.7.130:31091,192.168.7.130:31092')
    # producer = Producer(topic)
    # producer.start()

    video_path = 0
    # video_path = 'rtsp://admin:L277BFCA@221.165.56.189:3000/cam/realmonitor?channel=1&subtype=1'
    # video_path = 'rtsp://admin:L245AA6D@221.165.56.189:3001/cam/realmonitor?channel=1&subtype=1'
    run_on_video(video_path, '', conf_thresh=0.5)