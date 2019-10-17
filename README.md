# [DSFD](https://arxiv.org/abs/1810.10220?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)


## introduction

A tensorflow implement dsfd, and there is something different with the origin paper.

It‘s a ssd-like object detect framework, but slightly different,
combines lots of tricks for face detection, such as dual-shot, dense anchor match, FPN,FEM and so on.

Now it is mainly optimised about face detection,and borrows some codes from other repos.

Ps, the code maybe not that clear, please be patience, i am still working on it, and forgive me for my poor english :)

** contact me if u have question 2120140200@mail.nankai.edu.cn **


The evaluation results are based on vgg with batchsize(2x6),pretrained model can be download from

+ [baidu disk](https://pan.baidu.com/s/1cUqnf9BwUVkCy0iT6EczKA) ( password ty4d )
+ [google drive](https://drive.google.com/open?id=1zCeXPdRPG6-4W8fqEl4uRD5ojHbDRH-o)

widerface val set

| Easy MAP | Medium MAP | hard MAP |
| :------: | :------: | :------: |
|  0.942 | 0.935 | 0.880 |


| fddb   |
| :------: | 
|  0.987 | 


## requirment

+ tensorflow1.12

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
2. download the imagenet pretrained vgg16 model from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
release it in the root dir,

3. but if u want to train from scratch set config.MODEL.pretrained_model=None,

4. if recover from a complet pretrained model  set config.MODEL.pretrained_model='yourmodel.ckpt',config.MODEL.continue_train=True

5. then, run:

   ```python train.py```
   
   and if u want to check the data when training, u could set vis in train_config.py as True


6. After training, run:

   ```python tools/auto_freeze.py```

   it reads the checkpoint file and produces detector.pb .


### evaluation
** fddb **
```
    python model_eval/fddb.py [--model [TRAINED_MODEL]] [--data_dir [DATA_DIR]]
                          [--split_dir [SPLIT_DIR]] [--result [RESULT_DIR]]
    --model              Path of the saved model,default ./model/detector.pb
    --data_dir           Path of fddb all images
    --split_dir          Path of fddb folds
    --result             Path to save fddb results
 ```
    
example `python model_eval/fddb.py --model model/detector.pb 
                                    --data_dir 'fddb/img/' 
                                    --split_dir fddb/FDDB-folds/ 
                                    --result 'result/' `
                                    
** widerface **
```
    python model_eval/wider.py [--model [TRAINED_MODEL]] [--data_dir [DATA_DIR]]
                           [--result [RESULT_DIR]]
    --model              Path of the saved model,default ./model/detector.pb
    --data_dir           Path of WIDER
    --result             Path to save WIDERface results
 ```
example `python model_eval/wider.py --model model/detector.pb 
                                    --data_dir 'WIDER/WIDER_val/' 
                                    --result 'result/' `


### visualization
![A demo](https://github.com/610265158/DSFD-tensorflow/blob/master/figures/res_screenshot_11.05.2019.png)

(caution: i dont know where the demo picture comes from, if u think it's a tort, i would like to delete it.)

if u get a trained model, run `python tools/auto_freeze.py`, it will read the checkpoint file in ./model, and produce detector.pb, then

`python vis.py`

u can check th code in vis.py to make it runable, it's simple.




### details
#### anchor

if u like to show the anchor stratergy, u could simply run :

`python lib/core/anchor/anchor.py`


it will draw the anchor one by one,



### References
[DSFD: Dual Shot Face Detector](https://arxiv.org/abs/1810.10220?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)
