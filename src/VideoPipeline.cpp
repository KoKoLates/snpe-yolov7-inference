#include "VideoPipeline.hpp"

VideoPipeline::VideoPipeline(const Json::Value &value) {
    m_input_pipeline  = value["pipeline-config"]["input"].asString();
    m_output_pipeline = value["pipeline-config"]["output"].asString();
}

VideoPipeline::~VideoPipeline() {
    m_cap.release();
    m_out.release();
}

bool
VideoPipeline::create(const std::string &model_path) {
    m_detector = std::shared_ptr<yolov7::Detector>(new yolov7::Detector());
    m_detector->init(model_path);
    if (!m_detector->isInit()) {
        std::cerr << "[error] failed to initialize detector." << std::endl;
        return false; 
    }

    m_cap.open(m_input_pipeline,  cv::CAP_GSTREAMER);
    m_out.open(m_output_pipeline, cv::CAP_GSTREAMER, 20, cv::Size(1280, 720));
    if (!m_cap.isOpened() || !m_out.isOpened()) {
        std::cerr << "[error] failed to initialize video capture or writer" << std::endl;
        return false;
    }
    return true;
}

void
VideoPipeline::start() {
    std::thread captureThread(&VideoPipeline::produce, this);
    std::thread processThread(&VideoPipeline::consume, this);

    captureThread.join();
    processThread.join();
}

void
VideoPipeline::produce() {
    cv::Mat frame;
    while (true) {
        m_cap.read(frame);
        if (frame.empty()) break;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            while (m_queue.size() > m_max_size) {
                m_queue.pop();
            }
            m_queue.push(frame);
        }
    }
}

void 
VideoPipeline::consume() {
    while (true) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_queue.empty()) {
                frame = m_queue.front();
                m_queue.pop();
            }
        }

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // inference
        m_detector->detect(frame);
        m_out.write(frame);
    }
}