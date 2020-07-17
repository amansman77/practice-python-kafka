import numpy as np
from PIL import Image
import sys
import time
import os
import argparse
import rapidjson as json
import struct
import cv2
from geventhttpclient import HTTPClient
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression

feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
anchors_exp = np.expand_dims(anchors, axis=0)
id2class = {0: 'Mask', 1: 'NoMask'}

def preprocess(img):
  target_shape = (260, 260)
  resized_img = cv2.resize(img, target_shape)
  image_np = resized_img / 255.0
  image_exp = np.expand_dims(image_np, axis=0)
  return image_exp

def _get_inference_request(inputs, outputs=None):
  infer_request = {}
  parameters = {}

  infer_request['inputs'] = [
    {"shape":[1,260,260,3], "data": [val.item() for val in inputs.flatten()], "datatype":"FP32","name":"data" }
  ]

  request_body = json.dumps(infer_request)
  return request_body, None

def infer(img, uri, conf_thresh=0.5, iou_thresh=0.4, draw_result=True):

  processed_data = preprocess(img)

  request_body, json_size = _get_inference_request(processed_data, None)

  #serialized = [val.item() for val in processed_data]
  result = _post(request_uri=uri, request_body=request_body, headers=None, query_params=None)
  response = json.loads(result.read().decode('utf-8'))

  return response['outputs']

def _post(request_uri, request_body, headers, query_params):
        """Issues the POST request to the server

        Parameters
        ----------
        request_uri: str
            The request URI to be used in POST request.
        request_body: str
            The body of the request
        headers: dict
            Additional HTTP headers to include in the request.
        query_params: dict
            Optional url query parameters to use in network
            transaction.

        Returns
        -------
        geventhttpclient.response.HTTPSocketPoolResponse
            The response from server.
        """
        _client_stub = HTTPClient.from_url(
                    request_uri,
                    concurrency=1,
                    connection_timeout=60.0,
                    network_timeout = 60.0
                )
        if query_params is not None:
            request_uri = request_uri + "?" + _get_query_string(query_params)

        #print("POST {}, headers {}\n{}".format(request_uri, headers,
        #                                       request_body))

        if headers is not None:
            response = _client_stub.post(request_uri=request_uri,
                                              body=request_body,
                                              headers=headers)
        else:
            response = _client_stub.post(request_uri=request_uri,
                                              body=request_body)
        return response

def postprocess(img, response_data, height, width, conf_thresh=0.5, iou_thresh=0.4, draw_result=True):
  output_info = []

  y_cls_output = response_data[0]['data']
  y_bboxes_output = response_data[1]['data']

  y_bboxes_output = np.array(y_bboxes_output)
  y_cls_output = np.array(y_cls_output)
  y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
  y_cls = y_cls_output[0]

  bbox_max_scores = np.max(y_cls, axis=1)
  bbox_max_score_classes = np.argmax(y_cls, axis=1)

  keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                bbox_max_scores,
                                                conf_thresh=conf_thresh,
                                                iou_thresh=iou_thresh
                                                )
  for idx in keep_idxs:
    conf = float(bbox_max_scores[idx])
    class_id = bbox_max_score_classes[idx]
    bbox = y_bboxes[idx]

    xmin = max(0, int(bbox[0] * width))
    ymin = max(0, int(bbox[1] * height))
    xmax = min(int(bbox[2] * width), width)
    ymax = min(int(bbox[3] * height), height)
    if draw_result:
      if class_id == 0:
        color = (0, 255, 0)
      else:
        color = (255, 0, 0)
      cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
      cv2.putText(img, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
    output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

  img_output = Image.fromarray(img)
  return img_output, output_info

def run_on_video(video_path, output_video_name, conf_thresh):
  cap = cv2.VideoCapture(video_path)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  fps = cap.get(cv2.CAP_PROP_FPS)
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
  total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  if not cap.isOpened():
    raise ValueError("Video open failed.")
    return
  status = True
  idx = 0
  i =0
  while status:
    start_stamp = time.time()
    status, img_raw = cap.read()
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    read_frame_stamp = time.time()
    if (status):
      height, width, _ = img_raw.shape
      response = infer(img=img_raw, uri='http://192.168.7.122:32199/v2/models/model_2/infer')
      img_output, output_info = postprocess(img_raw, response, height, width)
      cv2.imshow('image', img_raw[:, :, ::-1])
      cv2.waitKey(1)
      inference_stamp = time.time()
      # writer.write(img_raw)
      write_frame_stamp = time.time()
      idx += 1
      # print("%d of %d" % (idx, total_frames))
      print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                             inference_stamp - read_frame_stamp,
                                                             write_frame_stamp - inference_stamp))
      #img_raw.save('test' + i + '.png')
      i = i + 1;
if __name__ ==  '__main__':
  output_info = []

  parser = argparse.ArgumentParser(description="Face Mask Detection")
  parser.add_argument('--img-mode', type=int, default=0, help='set 1 to run on image, 0 to run on video.')
  parser.add_argument('--img-path', type=str, default='demo2.jpg', help='path to your image.')
  parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')

  args = parser.parse_args()
  if args.img_mode:
    imgPath = args.img_path
    start_time = time.time()
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    response = infer(img=img, uri='http://192.168.7.130:32199/v2/models/model_2/infer')
    height, width, _ = img.shape
    infer_time = time.time()
    print("infer time: ", infer_time - start_time)
    img_output, output_info = postprocess(img, response, height, width)
    complete_time = time.time()
    print("complete time: ", complete_time - start_time)
    img_output.show()
    #img_output.save('test.png')
  else:
    video_path = args.video_path
    if args.video_path == '0':
      video_path = 0
    run_on_video(video_path, '', conf_thresh=0.5)
