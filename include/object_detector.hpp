#ifndef OBJECT_DETECTOR_HPP_
#define OBJECT_DETECTOR_HPP_

#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "snpe_pipeline.hpp"

namespace yolov7 {
struct BBox {
    cv::Rect bbox;
    int label = -1;
    float confidence = -1.f;
};

class Detector {
public:
    Detector();
    ~Detector();

    bool init(const std::string &model_path);
    bool is_init() { return m_init; }

    bool detect(cv::Mat &image, std::vector<BBox> &results);

private:
    bool m_init = false;
    std::unique_ptr<SNPEPipeline> snpe_task;
    
    bool preprocess(cv::Mat &image);
    bool postprocess(cv::Mat &image, std::vector<BBox> &results);

    std::vector<BBox> nms(std::vector<BBox> win_list, const float &nms_thres);
    float calculate_iou(const cv::Rect &a, const cv::Rect &b);
};

}


#endif // OBJECT_DETECTOR_HPP_