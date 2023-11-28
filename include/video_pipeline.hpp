#ifndef VIDEO_PIPELINE_HPP_
#define VIDEO_PIPELINE_HPP_

#include <queue>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "object_detector.hpp"

class VideoPipeline {
public:
    VideoPipeline(const std::string &input, const std::string &output);
    ~VideoPipeline();

    bool create(const std::string &model_path);
    unsigned int getMaxSize() const;
    unsigned int getCurSize() const;

    void start();

private:
    cv::VideoCapture m_cap;
    cv::VideoWriter  m_out;

    std::mutex m_mutex;
    std::queue<cv::Mat> m_queue;
    std::string input_pipeline, output_pipeline;
    std::shared_ptr<yolov7::Detector> m_detector;

    const unsigned int m_maxsize = 35;

    void produce();
    void consume();
};


#endif // VIDEO_PIPELINE_HPP_