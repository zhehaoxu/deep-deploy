#include "utils.h"
#include <MNN/Interpreter.hpp>
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

  cv::Mat img = cv::imread(argv[2]);
  int height = img.rows;
  int width = img.cols;
  std::vector<int> shape{1, 3, height, width};
  net->resizeTensor(input, shape);
  net->resizeSession(session);

  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  std::vector<float> image;

  // convert nhwc layout to nchw
  for (size_t i = 0; i < 3; i++) {
    int index = 0;
    for (size_t r = 0; r < height; r++) {
      for (size_t c = 0; c < width; c++) {
        float val = *(img.data + i + 3 * index);
        image.push_back(val / 255);
        index++;
      }
    }
  }

  auto nchwTensor = Tensor::create(input->shape(), input->getType(),
                                   image.data(), Tensor::CAFFE);

  input->copyFromHostTensor(nchwTensor);

  net->runSession(session);

  auto values = output->host<float>();
  int out_height = output->shape()[2];
  int out_width = output->shape()[3];
  int channel_chunk = 8; // just don't known why

  cv::Mat dst(out_width, out_height, CV_8UC3);
  for (size_t i = 0; i < out_height; i++) {
    for (size_t j = 0; j < out_width; j++) {
      int idx = (i * out_width + j) * channel_chunk;

      int r = clamp(values[idx] * 255., 0, 255);
      int g = clamp(values[idx + 1] * 255., 0, 255);
      int b = clamp(values[idx + 2] * 255., 0, 255);

      dst.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
    }
  }
  cv::imwrite("output.jpg", dst);
  return 0;
}
