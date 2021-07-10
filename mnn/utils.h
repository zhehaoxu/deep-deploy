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

// BGR order, color from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/colormap.py
// we may find a better way to get colormap
std::vector<cv::Scalar> COLOR_MAP{
    cv::Scalar(188, 113, 0),   cv::Scalar(24, 82, 216),
    cv::Scalar(31, 176, 236),  cv::Scalar(141, 46, 125),
    cv::Scalar(47, 171, 118),  cv::Scalar(237, 189, 76),
    cv::Scalar(46, 19, 161),   cv::Scalar(76, 76, 76),
    cv::Scalar(153, 153, 153), cv::Scalar(0, 0, 255),
    cv::Scalar(0, 127, 255),   cv::Scalar(0, 190, 190),
    cv::Scalar(0, 255, 0),     cv::Scalar(255, 0, 0),
    cv::Scalar(255, 0, 170),   cv::Scalar(0, 84, 84),
    cv::Scalar(0, 170, 84),    cv::Scalar(0, 255, 84),
    cv::Scalar(0, 84, 170),    cv::Scalar(0, 170, 170),
    cv::Scalar(0, 255, 170),   cv::Scalar(0, 84, 255),
    cv::Scalar(0, 170, 255),   cv::Scalar(0, 255, 255),
    cv::Scalar(127, 84, 0),    cv::Scalar(127, 170, 0),
    cv::Scalar(127, 255, 0),   cv::Scalar(127, 0, 84),
    cv::Scalar(127, 84, 84),   cv::Scalar(127, 170, 84),
    cv::Scalar(127, 255, 84),  cv::Scalar(127, 0, 170),
    cv::Scalar(127, 84, 170),  cv::Scalar(127, 170, 170),
    cv::Scalar(127, 255, 170), cv::Scalar(127, 0, 255),
    cv::Scalar(127, 84, 255),  cv::Scalar(127, 170, 255),
    cv::Scalar(127, 255, 255), cv::Scalar(255, 84, 0),
    cv::Scalar(255, 170, 0),   cv::Scalar(255, 255, 0),
    cv::Scalar(255, 0, 84),    cv::Scalar(255, 84, 84),
    cv::Scalar(255, 170, 84),  cv::Scalar(255, 255, 84),
    cv::Scalar(255, 0, 170),   cv::Scalar(255, 84, 170),
    cv::Scalar(255, 170, 170), cv::Scalar(255, 255, 170),
    cv::Scalar(255, 0, 255),   cv::Scalar(255, 84, 255),
    cv::Scalar(255, 170, 255), cv::Scalar(0, 0, 84),
    cv::Scalar(0, 0, 127),     cv::Scalar(0, 0, 170),
    cv::Scalar(0, 0, 212),     cv::Scalar(0, 0, 255),
    cv::Scalar(0, 42, 0),      cv::Scalar(0, 84, 0),
    cv::Scalar(0, 127, 0),     cv::Scalar(0, 170, 0),
    cv::Scalar(0, 212, 0),     cv::Scalar(0, 255, 0),
    cv::Scalar(42, 0, 0),      cv::Scalar(84, 0, 0),
    cv::Scalar(127, 0, 0),     cv::Scalar(170, 0, 0),
    cv::Scalar(212, 0, 0),     cv::Scalar(255, 0, 0),
    cv::Scalar(0, 0, 0),       cv::Scalar(36, 36, 36),
    cv::Scalar(218, 218, 218), cv::Scalar(255, 255, 255),
    cv::Scalar(0, 212, 0),     cv::Scalar(0, 255, 0),
    cv::Scalar(42, 0, 0),      cv::Scalar(84, 0, 0),
    cv::Scalar(127, 0, 0),     cv::Scalar(170, 0, 0),
};

std::unordered_map<int, std::string> COCO{
    {0, "person"},         {1, "bicycle"},       {2, "car"},
    {3, "motorbike"},      {4, "aeroplane"},     {5, "bus"},
    {6, "train"},          {7, "truck"},         {8, "boat"},
    {9, "traffic light"},  {10, "fire hydrant"}, {11, "stop sign"},
    {12, "parking meter"}, {13, "bench"},        {14, "bird"},
    {15, "cat"},           {16, "dog"},          {17, "horse"},
    {18, "sheep"},         {19, "cow"},          {20, "elephant"},
    {21, "bear"},          {22, "zebra"},        {23, "giraffe"},
    {24, "backpack"},      {25, "umbrella"},     {26, "handbag"},
    {27, "tie"},           {28, "suitcase"},     {29, "frisbee"},
    {30, "skis"},          {31, "snowboard"},    {32, "sports ball"},
    {33, "kite"},          {34, "baseball bat"}, {35, "baseball glove"},
    {36, "skateboard"},    {37, "surfboard"},    {38, "tennis racket"},
    {39, "bottle"},        {40, "wine glass"},   {41, "cup"},
    {42, "fork"},          {43, "knife"},        {44, "spoon"},
    {45, "bowl"},          {46, "banana"},       {47, "apple"},
    {48, "sandwich"},      {49, "orange"},       {50, "broccoli"},
    {51, "carrot"},        {52, "hot dog"},      {53, "pizza"},
    {54, "donut"},         {55, "cake"},         {56, "chair"},
    {57, "sofa"},          {58, "pottedplant"},  {59, "bed"},
    {60, "diningtable"},   {61, "toilet"},       {62, "tvmonitor"},
    {63, "laptop"},        {64, "mouse"},        {65, "remote"},
    {66, "keyboard"},      {67, "cell phone"},   {68, "microwave"},
    {69, "oven"},          {70, "toaster"},      {71, "sink"},
    {72, "refrigerator"},  {73, "book"},         {74, "clock"},
    {75, "vase"},          {76, "scissors"},     {77, "teddy bear"},
    {78, "hair drier"},    {79, "toothbrush"}};

float sigmoid(float x) { return 1. / (1. + std::exp(x * (-1.))); }

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
              const std::unordered_map<int, std::string> &labels = {}) {
  for (size_t i = 0; i < boxes.size(); i++) {
    cv::rectangle(image, cv::Point(int(boxes[i][0]), int(boxes[i][1])),
                  cv::Point(int(boxes[i][2]), int(boxes[i][3])),
                  COLOR_MAP[boxes[i][5]]);
    if (show_label) {
      int prob = int(boxes[i][4] * 100 + 0.5);
      std::string text = labels.at(boxes[i][5]) + ":" +
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
                    COLOR_MAP[boxes[i][5]], -1);
      cv::Point origin;
      origin.x = int(boxes[i][0]);
      origin.y = int(boxes[i][1]) + text_size.height + 2;
      cv::putText(image, text, origin, font, font_scale, cv::Scalar(0, 0, 0),
                  thickness, cv::LINE_AA);
    }
  }
}