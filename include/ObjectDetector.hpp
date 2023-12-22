#ifndef OBJECT_DETECTOR_HPP_
#define OBJECT_DETECTOR_HPP_

#include <string>
#include <vector>
#include <iostream>

#include "SNPEPipeline.hpp"
#include "opencv2/opencv.hpp"

namespace yolov7 {

struct Object {
    cv::Rect bbox;
    int label = -1;
    float confidence = -1.f;
};

struct Config {
    cv::Size frame;
    cv::Size target;
    int classes = -1;
    const std::string model_path;
};

class Detector {
public:
    Detector() {}

    bool init(const std::string &model_path);
    bool isInit() { return m_isInit; }

    bool detect(cv::Mat &frame);

private:
    bool m_isInit = false;
    std::unique_ptr<snpe::SNPEPipeline> m_snpe_task;

    bool preprocess(cv::Mat &frame);
    bool postprocess(cv::Mat &frame);

    std::vector<Object> nms(std::vector<Object> win_lis, const float &nms_thres);
    float calcIOU(const cv::Rect &a, const cv::Rect &b);
};

}


#endif // OBJECT_DETECTOR_HPP_