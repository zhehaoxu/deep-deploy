#pragma once

#include <MNN/Tensor.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using namespace MNN;

// color from
// https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/core/evaluation/class_names.py
std::vector<std::vector<int>> COLOR_MAP{
    {120, 120, 120}, {180, 120, 120}, {6, 230, 230},   {80, 50, 50},
    {4, 200, 3},     {120, 120, 80},  {140, 140, 140}, {204, 5, 255},
    {230, 230, 230}, {4, 250, 7},     {224, 5, 255},   {235, 255, 7},
    {150, 5, 61},    {120, 120, 70},  {8, 255, 51},    {255, 6, 82},
    {143, 255, 140}, {204, 255, 4},   {255, 51, 7},    {204, 70, 3},
    {0, 102, 200},   {61, 230, 250},  {255, 6, 51},    {11, 102, 255},
    {255, 7, 71},    {255, 9, 224},   {9, 7, 230},     {220, 220, 220},
    {255, 9, 92},    {112, 9, 255},   {8, 255, 214},   {7, 255, 224},
    {255, 184, 6},   {10, 255, 71},   {255, 41, 10},   {7, 255, 255},
    {224, 255, 8},   {102, 8, 255},   {255, 61, 6},    {255, 194, 7},
    {255, 122, 8},   {0, 255, 20},    {255, 8, 41},    {255, 5, 153},
    {6, 51, 255},    {235, 12, 255},  {160, 150, 20},  {0, 163, 255},
    {140, 140, 140}, {250, 10, 15},   {20, 255, 0},    {31, 255, 0},
    {255, 31, 0},    {255, 224, 0},   {153, 255, 0},   {0, 0, 255},
    {255, 71, 0},    {0, 235, 255},   {0, 173, 255},   {31, 0, 255},
    {11, 200, 200},  {255, 82, 0},    {0, 255, 245},   {0, 61, 255},
    {0, 255, 112},   {0, 255, 133},   {255, 0, 0},     {255, 163, 0},
    {255, 102, 0},   {194, 255, 0},   {0, 143, 255},   {51, 255, 0},
    {0, 82, 255},    {0, 255, 41},    {0, 255, 173},   {10, 0, 255},
    {173, 255, 0},   {0, 255, 153},   {255, 92, 0},    {255, 0, 255},
    {255, 0, 245},   {255, 0, 102},   {255, 173, 0},   {255, 0, 20},
    {255, 184, 184}, {0, 31, 255},    {0, 255, 61},    {0, 71, 255},
    {255, 0, 204},   {0, 255, 194},   {0, 255, 82},    {0, 10, 255},
    {0, 112, 255},   {51, 0, 255},    {0, 194, 255},   {0, 122, 255},
    {0, 255, 163},   {255, 153, 0},   {0, 255, 10},    {255, 112, 0},
    {143, 255, 0},   {82, 0, 255},    {163, 255, 0},   {255, 235, 0},
    {8, 184, 170},   {133, 0, 255},   {0, 255, 92},    {184, 0, 255},
    {255, 0, 31},    {0, 184, 255},   {0, 214, 255},   {255, 0, 112},
    {92, 255, 0},    {0, 224, 255},   {112, 224, 255}, {70, 184, 160},
    {163, 0, 255},   {153, 0, 255},   {71, 255, 0},    {255, 0, 163},
    {255, 204, 0},   {255, 0, 143},   {0, 255, 235},   {133, 255, 0},
    {255, 0, 235},   {245, 0, 255},   {255, 0, 122},   {255, 245, 0},
    {10, 190, 212},  {214, 255, 0},   {0, 204, 255},   {20, 0, 255},
    {255, 255, 0},   {0, 153, 255},   {0, 41, 255},    {0, 255, 204},
    {41, 0, 255},    {41, 255, 0},    {173, 0, 255},   {0, 245, 255},
    {71, 0, 255},    {122, 0, 255},   {0, 255, 184},   {0, 92, 255},
    {184, 255, 0},   {0, 133, 255},   {255, 214, 0},   {25, 194, 194},
    {102, 255, 0},   {92, 0, 255}};

std::vector<std::string> COCO{
    "person",        "bicycle",       "car",           "motorbike",
    "aeroplane",     "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "sofa",          "pottedplant",   "bed",
    "diningtable",   "toilet",        "tvmonitor",     "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush"};

std::vector<std::string> VOC{
    "background", "aeroplane",   "bicycle", "bird",  "boat",
    "bottle",     "bus",         "car",     "cat",   "chair",
    "cow",        "diningtable", "dog",     "horse", "motorbike",
    "person",     "pottedplant", "sheep",   "sofa",  "train",
    "tvmonitor"};

std::vector<std::string> CITYSCAPES{
    "road",    "sidewalk", "building", "wall",         "fence",
    "pole",    "traffic",  "light",    "traffic sign", "vegetation",
    "terrain", "sky",      "person",   "rider",        "car",
    "truck",   "bus",      "train",    "motorcycle",   "bicycle"};

float sigmoid(float x) { return 1. / (1. + std::exp(x * (-1.))); }

int clamp(float value, float min, float max) {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

void nc4hw4_to_nhwc(Tensor *tensor, int split, std::vector<float> &dst) {
  int channel = tensor->shape()[1];
  int height = tensor->shape()[2];
  int width = tensor->shape()[3];
  auto values = tensor->host<float>();
  int groups = int(std::ceil(1. * channel / split));

  for (size_t r = 0; r < height; r++) {
    for (size_t c = 0; c < width; c++) {
      int index = 0;
      for (size_t k = 0; k < groups; k++) {
        int idx = k * width * height * split + (r * width + c) * split;
        for (size_t d = 0; d < split; d++) {
          if (index >= channel)
            continue;
          dst.push_back(values[idx + d]);
          index++;
        }
      }
    }
  }
}

float compute_iou(std::vector<float> &box1, std::vector<float> &box2) {
  float box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
  float box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

  float hor =
      std::max(std::min(box1[2], box2[2]) - std::max(box1[0], box2[0]), 0.f);
  float ver =
      std::max(std::min(box1[3], box2[3]) - std::max(box1[1], box2[1]), 0.f);
  float inter_area = hor * ver;
  float iou = inter_area / (box1_area + box2_area - inter_area);
  return iou;
}

void nms(std::vector<std::vector<float>> &src_boxes,
         std::vector<std::vector<float>> &dst_boxes,
         std::unordered_set<int> &classes, float iou_threshold) {
  auto iter = classes.begin();
  for (; iter != classes.end(); iter++) {
    std::vector<std::vector<float>> cls_boxes;
    for (size_t i = 0; i < src_boxes.size(); i++) {
      if (int(src_boxes[i][5] == *iter)) {
        cls_boxes.push_back(src_boxes[i]);
      }
    }

    std::sort(cls_boxes.begin(), cls_boxes.end(),
              [](std::vector<float> &a, std::vector<float> &b) {
                return a[4] > b[4];
              });

    while (cls_boxes.size() > 0) {
      std::vector<std::vector<float>> remain_boxes;
      auto best_box = cls_boxes[0];
      dst_boxes.push_back(best_box);
      for (size_t i = 1; i < cls_boxes.size(); i++) {
        // compute iou
        float iou = compute_iou(best_box, cls_boxes[i]);
        if (iou < iou_threshold) {
          remain_boxes.push_back(cls_boxes[i]);
        }
      }
      cls_boxes = remain_boxes;
    }
  }
}

void draw_box(cv::Mat &image, std::vector<std::vector<float>> &boxes,
              bool show_label = true,
              const std::vector<std::string> &labels = {}) {
  for (size_t i = 0; i < boxes.size(); i++) {
    std::vector<int> color_ = COLOR_MAP[boxes[i][5]];
    cv::rectangle(image, cv::Point(int(boxes[i][0]), int(boxes[i][1])),
                  cv::Point(int(boxes[i][2]), int(boxes[i][3])),
                  cv::Scalar(color_[2], color_[1], color_[0]));
    if (show_label) {
      int prob = int(boxes[i][4] * 100 + 0.5);
      std::string text = labels[boxes[i][5]] + ":" +
                         std::to_string(prob / 100) + "." +
                         std::to_string(prob % 100);
      int font = cv::FONT_HERSHEY_SIMPLEX;
      double font_scale = 0.8;
      int thickness = 1;
      int baseline;
      cv::Size text_size =
          cv::getTextSize(text, font, font_scale, thickness, &baseline);

      cv::rectangle(image, cv::Point(int(boxes[i][0]), int(boxes[i][1])),
                    cv::Point(int(boxes[i][0]) + text_size.width,
                              int(boxes[i][1]) + text_size.height + 4),
                    cv::Scalar(color_[2], color_[1], color_[0]), -1);
      cv::Point origin;
      origin.x = int(boxes[i][0]);
      origin.y = int(boxes[i][1]) + text_size.height + 2;
      cv::putText(image, text, origin, font, font_scale, cv::Scalar(0, 0, 0),
                  thickness, cv::LINE_AA);
    }
  }
}