# 2019_ML_AI_workshop_sample_config

During the workshop, we will train an article detection from a 1910s newspaper page. 

you will find all the config files you would need for the workshop. 

#

`faster_rcnn_inception_resnet_v2.config`

source: https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/faster_rcnn_inception_resnet_v2_atrous_coco.config 

search for "update" and you will find all the fields that we will be updating for traiing a basic model. 

`faster_rcnn_inception_resnet_v2_sample.config` is the one we use during the workshop

#

`test_model_tensorflow_serving_rest.py` 

It is an exmaple of sending <b>REST API call to Tensorflow Serving API with a newspaper page image</b>. 

source: https://www.tensorflow.org/tfx/serving/tutorials/Serving_REST_simple 

to start tensorflow serving: 

<i>tensorflow_model_server  --port=8500 --rest_api_port=8501 --model_config_file=/home/ec2-user/sample_model/sample_model.config > tensorflowserving.log </i>

`test_model_tensorflow_serving_rest_sample.txt` is the output from python shell. This can be improved to be a function on your own time. 

#

`test1_output.png` is a sample output from the pre-train model.

#

`category.bptxt` is a ProtoBuf file you will find this file's path in the  `faster_rcnn_inception_resnet_v2_sample.config`