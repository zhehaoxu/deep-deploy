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

  int width = input->width();
  int height = input->height();
  int channel = input->channel();
  int size = width * height;

  cv::Mat img = cv::imread(argv[2]);
  cv::resize(img, img, cv::Size(width, height));
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  std::vector<float> image;
  auto nchwTensor = new Tensor(input, Tensor::CAFFE);
  // convert nhwc layout to nchw
  for (size_t i = 0; i < channel; i++) {
    for (size_t j = 0; j < size; j++) {
      float value = *(img.data + j * channel + i);
      nchwTensor->host<float>()[size * i + j] = value / 128.0 - 1.0;
    }
  }

  input->copyFromHostTensor(nchwTensor);

  net->runSession(session);
  std::vector<std::pair<int, float>> label;
  auto values = output->host<float>();
  for (int i = 0; i < output->elementSize(); ++i) {
    label.push_back(std::make_pair(i, values[i]));
  }

  std::sort(label.begin(), label.end(),
            [](std::pair<int, float> &a, std::pair<int, float> &b) {
              return a.second > b.second;
            });

  int topk = 5;
  for (size_t i = 0; i < topk; i++) {
    std::cout << "class: " << label[i].first << " prob: " << label[i].second
              << std::endl;
  }

  return 0;
}
