
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_folder = './model'
checkpoint = tf.train.get_checkpoint_state(model_folder)

##input_checkpoint
input_checkpoint = checkpoint.model_checkpoint_path
##input_graph
input_meta_graph = input_checkpoint + '.meta'

##output_node_names
output_node_names='tower_0/images,tower_0/boxes,tower_0/scores,tower_0/labels,tower_0/num_detections,training_flag'

#output_graph
output_graph='./model/detector.pb'


print('excuted')

command="python tools/freeze.py --input_checkpoint %s --input_meta_graph %s --output_node_names %s --output_graph %s"\
%(input_checkpoint,input_meta_graph,output_node_names,output_graph)
os.system(command)