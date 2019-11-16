# [DSFD](https://arxiv.org/abs/1810.10220?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)


## introduction

A tensorflow2 implement dsfd, and there is something different with the origin paper.

Now it is mainly optimised about face detection,and borrows some codes from other repos.

** contact me if u have question 2120140200@mail.nankai.edu.cn **





A light pretrained model can be download from,
if you want the vgg_dsfd, please retrain or switch to master

## pretrained model and performance

###### Lightnet_0.5  including tflite model, 
###### (time cost: mac i5-8279U@2.4GHz， tf2.0 15ms+， tflite 8ms+-,input shape 320x320)

+ [baidu disk](https://pan.baidu.com/s/1ZJZHJz8VFXahmwBptGQfiA) ( password yqst )
+ [google drive](https://drive.google.com/open?id=1ZZVA7QhwGWYJ-09KoU2iym90zqbrfTQH)




| model         |input_size |fddb      |model size|
| :------:      |:------:   |:------:  |:------:  |
| Lightnet_0.75|640x640     | 0.960    |800k+-|
| Lightnet_0.5 |640x640     | 0.953    |560k+-|
| Lightnet_0.5 |416x416     | 0.953    |560k+-|
| Lightnet_0.5 |320x320     | 0.936    |560k+-|

| model         |input_size  |wider easy|wider media |wider hard |
| :------:      |:------:     |:------:  | :------:  | :------:  | 
| Lightnet_0.75 |640x640      | 0.867    |0.806     |0.440      |
| Lightnet_0.5  |640x640      | 0.858    |0.796     |0.430      |
| Lightnet_0.5 |multiscale   | 0.861     |0.837     |0.726      |

ps the time cost not including nms, and flip test is used

## requirment

+ tensorflow2.0

+ tensorpack (for data provider)

+ opencv

+ python 3.6

## useage

### train
1. download widerface data from http://shuoyang1213.me/WIDERFACE/
and release the WIDER_train, WIDER_val and wider_face_split into ./WIDER, then run
```python prepare_wider_data.py```it will produce train.txt and val.txt
(if u like train u own data, u should prepare the data like this:
`...../9_Press_Conference_Press_Conference_9_659.jpg| 483(xmin),195(ymin),735(xmax),543(ymax),1(class) ......` 
one line for one pic, **caution! class should start from 1, 0 means bg**)

2. set config=lightnet_config in train_config.py   or check the train_config.py choose one

3. if recover from a pretrained model  set config.MODEL.pretrained_model='yourmodel' in the configs

4. then, run:

   ```python train.py```
   
   and if u want to check the data when training, u could set vis in train_config.py as True



### evaluation
** fddb **
```
    python model_eval/fddb.py [--model [TRAINED_MODEL]] [--data_dir [DATA_DIR]]
                          [--split_dir [SPLIT_DIR]] [--result [RESULT_DIR]]
    --model              Path of the saved model,default ./model/detector
    --data_dir           Path of fddb all images
    --split_dir          Path of fddb folds
    --result             Path to save fddb results
 ```
    
example `python model_eval/fddb.py --model model/detector 
                                    --data_dir 'fddb/img/' 
                                    --split_dir fddb/FDDB-folds/ 
                                    --result 'result/'
                                    --input_shape 640`
                                    
** widerface **
```
    python model_eval/wider.py [--model [TRAINED_MODEL]] [--data_dir [DATA_DIR]]
                           [--result [RESULT_DIR]] [--multiscale [test in multiscales  0-False 1-True]]
                           [--input_shape [input shape]]
    --model              Path of the saved model,default ./model/detector
    --data_dir           Path of WIDER
    --result             Path to save WIDERface results
 ```
example `python model_eval/wider.py --model model/detector 
                                    --data_dir 'WIDER/WIDER_val/' 
                                    --result 'result/'
                                    --multiscale 0
                                    --input_shape 640`




### convert tflite and quantization
Because tflite needs the input shape be fixed,
we should rebiuld the model first(refer to tools/convert_to_tflite.py),
set a suitable input shape, and set saved_model_dir='your model' 
then run 
`python tools/convert_to_tflite.py`


ps, if you want to do quantization when convert to tflite,
the postprocess should be done by yourself, please take care of it. 





### visualization
![A demo](https://github.com/610265158/DSFD-tensorflow/blob/master/figures/res_screenshot_11.05.2019.png)

(caution: i dont know where the demo picture comes from, if u think it's a tort, i would like to delete it.)


`python vis.py --model ./model/detector --img_dir ./FDDB` or

`python vis.py --model ./model/detector.tflite --img_dir ./FDDB`
u can check th code in vis.py to make it runable, it's simple.




### details
#### anchor

if u like to show the anchor stratergy, u could simply run :

`python lib/core/anchor/anchor.py`


it will draw the anchor one by one,



### References
[DSFD: Dual Shot Face Detector](https://arxiv.org/abs/1810.10220?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)
