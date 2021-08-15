#include "utils.h"
#include <MNN/Interpreter.hpp>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

using namespace MNN;
int main(int argc, char const *argv[]) {

  std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
  ScheduleConfig config;
  config.type = MNN_FORWARD_AUTO;

  auto session = net->createSession(config);
  auto input = net->getSessionInput(session, NULL);
  auto output = net->getSessionOutput(session, NULL);
  std::vector<int> shape{1, 3, 512, 512};
  net->resizeTensor(input, shape);
  net->resizeSession(session);
  int width = input->width();
  int height = input->height();
  int channel = input->channel();
  int size = width * height;

  cv::Mat img = cv::imread(argv[2]);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  int img_width = img.cols;
  int img_height = img.rows;

  float scale = std::min(1.0 * width / img_width, 1.0 * height / img_height);

  int scaled_width = int(scale * img_width);
  int scaled_height = int(scale * img_height);
  cv::Mat img_resized;
  cv::resize(img, img_resized, cv::Size(scaled_width, scaled_height));

  int pad_bottom = height - scaled_height;
  int pad_right = width - scaled_width;
  cv::copyMakeBorder(img_resized, img_resized, 0, pad_bottom, 0, pad_right,
                     cv::BORDER_CONSTANT, 0);

  //   std::vector<float> image;
  auto nchwTensor = new Tensor(input, Tensor::CAFFE);

  float mean[] = {123.675, 116.28, 103.53};
  float stddev[] = {58.395, 57.12, 57.375};

  // convert nhwc layout to nchw
  for (size_t i = 0; i < channel; i++) {
    for (size_t j = 0; j < size; j++) {
      float value = *(img_resized.data + j * channel + i);
      nchwTensor->host<float>()[size * i + j] = (value - mean[i]) / stddev[i];
    }
  }

  input->copyFromHostTensor(nchwTensor);

  net->runSession(session);
  cv::Mat map(scaled_height, scaled_width, CV_8UC3);
  auto values = output->host<int>();
  for (size_t i = 0; i < scaled_height; i++) {
    for (size_t j = 0; j < scaled_width; j++) {
      int cls_id = values[i * width + j];
      map.at<cv::Vec3b>(i, j) = cv::Vec3b(
          COLOR_MAP[cls_id][2], COLOR_MAP[cls_id][1], COLOR_MAP[cls_id][0]);
    }
  }
  cv::resize(map, map, cv::Size(img_width, img_height), 0, 0,
             cv::INTER_NEAREST_EXACT);
  cv::imwrite("output.jpg", map);
  return 0;
}
