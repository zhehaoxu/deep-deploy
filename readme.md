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

| model           | source                                                       | demo                                                         |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Mobilenet V1/V2 | [link](https://github.com/tensorflow/models/tree/master/research/slim) | [src](https://github.com/zhehaoxu/deep-deploy/blob/main/mnn/mobilenet.cc) |

```
./mobilenet mobilenet_v2.mnn cat.jpg
```

### object detection

| model   | source                                                   | demo                                                         |
| ------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| Yolo V3 | [link](https://github.com/YunYang1994/tensorflow-yolov3) | [src](https://github.com/zhehaoxu/deep-deploy/blob/main/mnn/yolov3.cc) |

```
./yolov3 yolov3.mnn road.jpg
```

`Note`: demo pass in lastest version



## tools

1. `frozen_graph.py`: to fix input shape or make partition.