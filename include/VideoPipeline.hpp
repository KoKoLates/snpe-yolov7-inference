#ifndef VIDEO_PIPELINE_HPP_
#define VIDEO_PIPELINE_HPP_

#include <queue>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>

#include "ObjectDetector.hpp"

class VideoPipeline {
public:
    VideoPipeline(const Json::Value &value);
    ~VideoPipeline();

    void start();
    bool create(const std::string &model_path);

private:
    cv::VideoCapture m_cap;
    cv::VideoWriter  m_out;

    std::mutex m_mutex; 
    std::queue<cv::Mat> m_queue;
    std::shared_ptr<yolov7::Detector> m_detector;

    std::string m_input_pipeline, m_output_pipeline;

    const unsigned int m_max_size = 35;

    void produce();
    void consume();
};

#endif // VIDEO_PIPELINE_HPP_