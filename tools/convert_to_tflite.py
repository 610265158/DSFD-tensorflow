import sys
sys.path.append('.')
import tensorflow as tf
import numpy as np
import time

from tools.tf_model_wraper import DSFDLite
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


input_shape=(320,320,3)

saved_model_dir='./xx'
saved_model_dir_for_lite=saved_model_dir+'lite_pre'

### we build the tflite version detect firstly, then save it

model=DSFDLite(input_shape)
model.load_weights(saved_model_dir+'/variables/variables')
tf.saved_model.save(model,saved_model_dir_for_lite)

print('the model rebuild over, ')

####
save_tf_model="converted_model.tflite"

model=tf.saved_model.load(saved_model_dir_for_lite)



concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

concrete_func.inputs[0].set_shape([1, *input_shape])

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])


tflite_model = converter.convert()

##write it down
open(save_tf_model, "wb").write(tflite_model)

# 加载 TFLite 模型并分配张量（tensor）。
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# 获取输入和输出张量。
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 使用随机数据作为输入测试 TensorFlow Lite 模型。
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

tflite_results = interpreter.get_tensor(output_details[0]['index'])
start=time.time()

for i in range(100):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

print((time.time()-start)/100.)
print(tflite_results)


tf_results = model.inference_fixed(tf.constant(input_data))
print(tf_results)
