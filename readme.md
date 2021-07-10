## deploy deep learning model with MNN.

## build
1. install opencv;

2. build MNN

3. build this project

   ```
   mkdir build;cd build
   cmake -DMNN="your MNN path" ..
   make
   ```

## use
### image classification

| model              | source                                                       | demo                                                         |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Mobilenet V1/V2/V3 | [link](https://github.com/tensorflow/models/tree/master/research/slim) | [src](https://github.com/zhehaoxu/deep-deploy/blob/main/mnn/mobilenet.cc) |

```
./mobilenet mobilenet_v2.mnn cat.jpg
```

Output:

```
class: 282 prob: 0.537347
class: 283 prob: 0.14923
class: 286 prob: 0.127461
class: 288 prob: 0.0312308
class: 284 prob: 0.0156367
```

### object detection

| model   | source                                                   | demo                                                         | note                    |
| ------- | -------------------------------------------------------- | ------------------------------------------------------------ | ----------------------- |
| Yolo V3 | [link](https://github.com/YunYang1994/tensorflow-yolov3) | [src](https://github.com/zhehaoxu/deep-deploy/blob/main/mnn/yolov3.cc) | pass in lastest version |
| Yolo V3 | [link](https://github.com/YunYang1994/tensorflow-yolov3) | [src](https://github.com/zhehaoxu/deep-deploy/blob/main/mnn/yolov3_hc.cc) | decode box by hard code |

```
./yolov3 yolov3.mnn image2.jpg
```

Output:

![detection](./images/output.jpg)



## tools

1. `frozen_graph.py`: to fix input shape or make partition.