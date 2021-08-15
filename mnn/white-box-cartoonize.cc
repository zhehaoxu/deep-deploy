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

  cv::Mat img = cv::imread(argv[2]);
  int height = img.rows;
  int width = img.cols;
  int h = height, w = width;
  if (std::min(height, width) > 720) {
    if (height > width) {
      h = int(720 * height / width);
      w = 720;
    } else {
      h = 720;
      w = int(720 * width / height);
    }
  }
  cv::resize(img, img, cv::Size(w, h));
  h = (h / 8) * 8;
  w = (w / 8) * 8;
  cv::Mat crop_img = img(cv::Rect(0, 0, w, h));
  cv::Mat dst;
  crop_img.copyTo(dst);
  cv::cvtColor(crop_img, crop_img, cv::COLOR_BGR2RGB);

  std::vector<int> shape{1, 3, h, w};
  net->resizeTensor(input, shape);
  net->resizeSession(session);

  std::vector<float> image;

  // convert nhwc layout to nchw
  for (size_t i = 0; i < 3; i++) {
    for (size_t r = 0; r < h; r++) {
      for (size_t c = 0; c < w; c++) {
        float val = crop_img.at<cv::Vec3b>(r, c)[i];
        image.push_back(val / 127.5 - 1);
      }
    }
  }
  auto nchwTensor = Tensor::create(input->shape(), input->getType(),
                                   image.data(), Tensor::CAFFE);
  input->copyFromHostTensor(nchwTensor);
  net->runSession(session);

  auto values = output->host<float>();
  for (size_t i = 0; i < h; i++) {
    for (size_t j = 0; j < w; j++) {
      int idx = (i * w + j) * 3;

      int r = clamp((values[idx] + 1) * 127.5, 0, 255);
      int g = clamp((values[idx + 1] + 1) * 127.5, 0, 255);
      int b = clamp((values[idx + 2] + 1) * 127.5, 0, 255);
      dst.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
    }
  }
  cv::imwrite("output.jpg", dst);
  return 0;
}
