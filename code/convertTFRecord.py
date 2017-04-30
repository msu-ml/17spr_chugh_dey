# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 22:54:38 2017

@author: tarang
"""
import tensorflow as tf
from tensorflow import gfile
import pandas as pd

data_video =[]
data_audio =[]
for label in range(40):
    data_video.append([])
    data_audio.append([])

srcPath = '/Users/tarang/Documents/MLProject/'
ftrType = 'frame_level/validate'
files = gfile.Glob(srcPath+ftrType+'/validate*.tfrecord')
print files
count=1;
for tfrecords_filename in files:
    print tfrecords_filename
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        video_id = (example.features.feature['video_id']
                                     .bytes_list
                                     .value[0])

        labels = (example.features.feature['labels']
                                    .int64_list.value[:])

        mean_rgb = (example.features.feature['mean_rgb']
                                      .float_list.value[:])

        mean_audio = (example.features.feature['mean_audio']
                                    .float_list
                                    .value[:])

        print video_id
        for l in labels:
            print l
        count=count+1
        if(count>128):
            break
        #for l in labels:
        #    if(l<40 and len(data_video[l])<2000):
        #        data_video[l].append(mean_rgb)
        #        data_audio[l].append(mean_audio)

'''
for i in range(40):
    destPath = srcPath + ftrType + '_csv/Class'+str(i)+'_vid.csv'
    my_df = pd.DataFrame(data_video[i])
    my_df.to_csv(destPath, index=False, header=False)
    #writer = csv.writer(destPath)
    #writer.writerow(data_video[i])
'''
