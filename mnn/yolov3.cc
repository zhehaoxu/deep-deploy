#include <MNN/Interpreter.hpp>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <utility>
#include <vector>

// TODO: refine code and draw pretty box
using namespace MNN;
int main(int argc, char const *argv[]) {
  std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
  ScheduleConfig config;
  config.type = MNN_FORWARD_AUTO;

  auto session = net->createSession(config);
  auto input = net->getSessionInput(session, NULL);
  // uncomment below if model shape is not fixed
  // std::vector<int> shape{1, 3, 416, 416};
  // net->resizeTensor(input, shape);
  // net->resizeSession(session);

  auto soutput = net->getSessionOutput(session, "pred_sbbox/concat_2");
  auto moutput = net->getSessionOutput(session, "pred_mbbox/concat_2");
  auto loutput = net->getSessionOutput(session, "pred_lbbox/concat_2");

  cv::Mat img = cv::imread(argv[2]);
  cv::Mat dst;
  img.copyTo(dst);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  int net_width = 416;
  int net_height = 416;
  int img_width = img.cols;
  int img_height = img.rows;

  float scale =
      std::min(1.0 * net_width / img_width, 1.0 * net_height / img_height);

  int scaled_width = int(scale * img_width);
  int scaled_height = int(scale * img_height);
  cv::Mat img_resized;
  cv::resize(img, img_resized, cv::Size(scaled_width, scaled_height));

  int pad_top = (net_height - scaled_height) / 2;
  int pad_left = (net_width - scaled_width) / 2;
  int pad_bottom = net_height - scaled_height - pad_top;
  int pad_right = net_width - scaled_width - pad_left;
  cv::copyMakeBorder(img_resized, img_resized, pad_top, pad_bottom, pad_left,
                     pad_right, cv::BORDER_CONSTANT, 0);

  std::vector<float> image;
  for (size_t i = 0; i < 3; i++) {
    int index = 0;
    for (size_t r = 0; r < net_height; r++) {
      for (size_t c = 0; c < net_width; c++) {
        float value = *(img_resized.data + i + 3 * index);
        image.push_back(value / 255.f);
        index++;
      }
    }
  }

  auto tensor = Tensor::create(input->shape(), input->getType(), image.data(),
                               Tensor::CAFFE);
  input->copyFromHostTensor(tensor);
  net->runSession(session);

  auto sbox = soutput->host<float>();
  auto mbox = moutput->host<float>();
  auto lbox = loutput->host<float>();

  std::vector<std::vector<float>> boxes;

  int sh = soutput->shape()[1];
  int sw = soutput->shape()[2];
  int sanchors = soutput->shape()[3];
  int spreds = soutput->shape()[4];
  for (size_t i = 0; i < sh * sw * sanchors; i++) {
    std::vector<float> box;
    for (size_t j = 0; j < spreds; j++) {
      box.push_back(sbox[j + i * spreds]);
    }
    boxes.push_back(box);
  }

  int mh = moutput->shape()[1];
  int mw = moutput->shape()[2];
  int manchors = moutput->shape()[3];
  int mpreds = moutput->shape()[4];
  for (size_t i = 0; i < mh * mw * manchors; i++) {
    std::vector<float> box;
    for (size_t j = 0; j < mpreds; j++) {
      box.push_back(mbox[j + i * mpreds]);
    }
    boxes.push_back(box);
  }

  int lh = loutput->shape()[1];
  int lw = loutput->shape()[2];
  int lanchors = loutput->shape()[3];
  int lpreds = loutput->shape()[4];
  for (size_t i = 0; i < lh * lw * lanchors; i++) {
    std::vector<float> box;
    for (size_t j = 0; j < lpreds; j++) {
      box.push_back(lbox[j + i * lpreds]);
    }
    boxes.push_back(box);
  }

  float score_threshold = 0.3;
  std::vector<std::vector<float>> after_boxes;
  std::unordered_set<int> classes;
  // ref
  // https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/utils.py
  for (size_t i = 0; i < boxes.size(); i++) {
    float x = boxes[i][0];
    float y = boxes[i][1];
    float w = boxes[i][2];
    float h = boxes[i][3];
    // (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    float xmin = x - w / 2.f;
    float ymin = y - h / 2.f;
    float xmax = x + w / 2.f;
    float ymax = y + h / 2.f;
    // (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    float xmin_org = (xmin - pad_left) / scale;
    float xmax_org = (xmax - pad_left) / scale;
    float ymin_org = (ymin - pad_top) / scale;
    float ymax_org = (ymax - pad_top) / scale;

    xmin_org = std::max(xmin_org, 0.f);
    ymin_org = std::max(ymin_org, 0.f);
    xmax_org = std::min(xmax_org, img_width - 1.f);
    ymax_org = std::min(ymax_org, img_height - 1.f);
    // clip some boxes those are out of range
    if (xmin_org >= xmax_org || ymin_org >= ymax_org) {
      continue;
    }

    int index = 0;
    float prob = 0.f;
    for (size_t k = 5; k < boxes[i].size(); k++) {
      if (boxes[i][k] > prob) {
        prob = boxes[i][k];
        index = k;
      }
    }
    // discard some boxes with low scores
    float score = boxes[i][4] * prob;
    if (score < score_threshold) {
      continue;
    }

    after_boxes.push_back(
        {xmin_org, ymin_org, xmax_org, ymax_org, score, index - 5.f});
    classes.emplace(index - 5);
  }

  float iou_threshold = 0.45;
  std::vector<std::vector<float>> best_boxes;
  auto iter = classes.begin();
  for (; iter != classes.end(); iter++) {
    std::vector<std::vector<float>> cls_boxes;
    for (size_t i = 0; i < after_boxes.size(); i++) {
      if (int(after_boxes[i][5] == *iter)) {
        cls_boxes.push_back(after_boxes[i]);
      }
    }

    std::sort(cls_boxes.begin(), cls_boxes.end(),
              [](std::vector<float> &a, std::vector<float> &b) {
                return a[4] > b[4];
              });

    while (cls_boxes.size() > 0) {
      std::vector<std::vector<float>> remain_boxes;
      auto best_box = cls_boxes[0];
      best_boxes.push_back(best_box);
      for (size_t i = 1; i < cls_boxes.size(); i++) {
        // compute iou
        float box1_area =
            (best_box[2] - best_box[0]) * (best_box[3] - best_box[1]);
        float box2_area = (cls_boxes[i][2] - cls_boxes[i][0]) *
                          (cls_boxes[i][3] - cls_boxes[i][1]);

        float hor = std::max(std::min(best_box[2], cls_boxes[i][2]) -
                                 std::max(best_box[0], cls_boxes[i][0]),
                             0.f);
        float ver = std::max(std::min(best_box[3], cls_boxes[i][3]) -
                                 std::max(best_box[1], cls_boxes[i][1]),
                             0.f);
        float inter_area = hor * ver;
        float iou = inter_area / (box1_area + box2_area - inter_area);
        if (iou < iou_threshold) {
          remain_boxes.push_back(cls_boxes[i]);
        }
      }
      cls_boxes = remain_boxes;
    }
  }
  // draw box
  for (size_t i = 0; i < best_boxes.size(); i++) {
    cv::rectangle(dst, cv::Point(int(best_boxes[i][0]), int(best_boxes[i][1])),
                  cv::Point(int(best_boxes[i][2]), int(best_boxes[i][3])),
                  cv::Scalar(0, 0, 255));
  }
  cv::imwrite("output.jpg", dst);
  return 0;
}
