import numpy as np
import pandas as pd

data_dir='../global-wheat-detection/train'
train_csv='../global-wheat-detection/train.csv'


ratio=0.9

train_data=pd.read_csv(train_csv)

print(train_data)


image_ids=list(set(train_data['image_id']))

train_list=image_ids[:int(ratio*len(image_ids))]
val_list=image_ids[int(ratio*len(image_ids)):]

klasses=set(train_data['source'])

train_file=open('train.txt', 'w')


for k,id in enumerate(train_list):

    bboxes=train_data[train_data['image_id']==id]

    cur_label_message=data_dir+'/'+str(id)+'.jpg|'

    for box in bboxes['bbox']:
        curbox=box[1:-1].split(',')
        cur_box_info=[float(x) for x in curbox]

        cur_box_info=" "+ str(cur_box_info[0])+',' + str(cur_box_info[1])+ ','+\
                     str(cur_box_info[0]+cur_box_info[2])+","+str(cur_box_info[1]+cur_box_info[3]) +',1'
        cur_label_message=cur_label_message+cur_box_info

    cur_label_message+='\n'
    train_file.write(cur_label_message)


val_file=open('val.txt', 'w')


for k,id in enumerate(val_list):

    bboxes=train_data[train_data['image_id']==id]

    cur_label_message=data_dir+'/'+str(id)+'.jpg|'

    for box in bboxes['bbox']:
        curbox=box[1:-1].split(',')
        cur_box_info=[float(x) for x in curbox]

        cur_box_info=" "+ str(cur_box_info[0])+',' + str(cur_box_info[1])+ ','+\
                     str(cur_box_info[0]+cur_box_info[2])+","+str(cur_box_info[1]+cur_box_info[3]) +',1'
        cur_label_message=cur_label_message+cur_box_info

    cur_label_message+='\n'
    val_file.write(cur_label_message)


