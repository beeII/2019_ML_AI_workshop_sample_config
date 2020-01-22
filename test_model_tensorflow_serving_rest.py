## https://www.tensorflow.org/tfx/serving/tutorials/Serving_REST_simple#make_rest_requests
## start model serving 
## tensorflow_model_server  --port=8500 --rest_api_port=8501 --model_config_file=/home/ec2-user/sample_model/sample_model.config > tensorflowserving.log
from __future__ import print_function
import requests
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import json 
import sys
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def process_image(image_path, output_path):
  headers = {"content-type": "application/json"}
  image = Image.open(image_path)
  image_np = load_image_into_numpy_array(image)
  image_np_expanded = np.expand_dims(image_np, axis=0)
  path_to_label="/home/ec2-user/Category.pdtxt"
  category_index = label_map_util.create_category_index_from_labelmap(path_to_label)
  data = json.dumps({"signature_name": "serving_default", "instances": image_np_expanded.tolist()})
  json_response = requests.post('http://localhost:8501/v1/models/sample_model:predict', data=data, headers=headers)
  predictions = json.loads(json_response.text)['predictions']
  output_dict={}
  output_dict['num_detections'] = int(predictions[0]['num_detections'])
  output_dict['detection_classes'] = np.array(predictions[0]['detection_classes']).astype(np.int64)
  output_dict['detection_boxes'] = np.array(predictions[0]['detection_boxes'])
  output_dict['detection_scores'] = np.array(predictions[0]['detection_scores'])

  vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
  image_size = (12, 8)
  plt.figure(figsize=image_size)
  plt.imsave(output_path, image_np)
  plt.close('all')


if __name__ == '__main__':
  if(len(sys.argv)>=3):
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    process_image(image_path, output_path)
  else:
    print("python file.py image_path output_path")


