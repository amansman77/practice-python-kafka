from kafka import KafkaProducer
import cv2
import time
import numpy as np

def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
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
            print ('[Send video] Topic: ' + topic)
            
topic = 'fmd-video-1'
producer = KafkaProducer(bootstrap_servers='192.168.7.130:31090,192.168.7.130:31091,192.168.7.130:31092')

# video_path = 0
video_path = 'rtsp://admin:L245AA6D@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1'
run_on_video(video_path, '', conf_thresh=0.5)