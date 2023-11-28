#include "video_pipeline.hpp"

VideoPipeline::VideoPipeline(const std::string &input, const std::string &output) {
    this->input_pipeline  = input;
    this->output_pipeline = output;
}

VideoPipeline::~VideoPipeline() {
    this->m_cap.release();
    this->m_out.release();
}

bool
VideoPipeline::create(const std::string &model_path) {
    m_detector = std::shared_ptr<yolov7::Detector>(new yolov7::Detector());
    m_detector->init(model_path);
    if (!m_detector->is_init()) {
        std::cerr << "[error] failed to initialize yolov7 detector" << std::endl;
        return false;
    }

    this->m_cap.open(input_pipeline);
    this->m_cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
    this->m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 768);

    this->m_out.open(output_pipeline, cv::CAP_GSTREAMER, 10, cv::Size(1024, 768));
    if (!this->m_cap.isOpened() || !this->m_out.isOpened()) {
        std::cout << "[error] video capture or writer failed to load." << std::endl;
        return false;
    }
    return true;
}

unsigned int
VideoPipeline::getMaxSize() const {
    return this->m_maxsize;
}

unsigned int
VideoPipeline::getCurSize() const {
    return this->m_queue.size();
}

void
VideoPipeline::start() {
    std::thread produce_thread(&VideoPipeline::produce, this);
    std::thread consume_thread(&VideoPipeline::consume, this);

    produce_thread.join();
    consume_thread.join();
}

void
VideoPipeline::produce() {
    cv::Mat frame;
    while (true) {
        this->m_cap.read(frame);
        if (frame.empty()) { break; }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            while (this->m_queue.size() > this->m_maxsize) {
                this->m_queue.pop();
            }
            this->m_queue.push(frame);
        }
    }
}

void
VideoPipeline::consume() {
    while (true) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!this->m_queue.empty()) {
                frame = this->m_queue.front();
                this->m_queue.pop();
            }
        }

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // process here
        std::vector<yolov7::BBox> results;
        m_detector->detect(frame, results);
        
        cv::resize(frame, frame, cv::Size(1024, 768));
        this->m_out.write(frame);
    }
}