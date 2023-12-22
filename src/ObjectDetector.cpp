#include "ObjectDetector.hpp"

namespace yolov7 {

bool
Detector::init(const std::string &model_path) {
    m_snpe_task = std::move(std::unique_ptr<snpe::SNPEPipeline>(new snpe::SNPEPipeline()));
    if (!m_snpe_task->init(model_path)) {
        std::cerr << "[error] failed to initialize snpe instance" << std::endl;
        return false;
    }
    m_isInit = m_snpe_task->isInit();
    return m_isInit;
}

bool
Detector::detect(cv::Mat &frame) {
    if (!preprocess(frame)) {
        std::cerr << "[error] failed to preprocess" << std::endl;
        return false;
    }
    if (!m_snpe_task->execute()) {
        std::cerr << "[error] failed to execute snpe model" << std::endl;
        return false;
    }
    if (!postprocess(frame)) {
        std::cerr << "[error] failed to postprocess" << std::endl;
        return false;
    }
    return true;
}

bool
Detector::preprocess(cv::Mat &frame) {
    cv::resize(frame, frame, cv::Size(416, 416), cv::INTER_LINEAR);

    std::vector<float> input_vec;
    for (float y = 0; y < 416; y++) {
        for (float x = 0; x < 416; x++) {
            cv::Vec3b value = frame.at<cv::Vec3b>(y, x);
            float r = static_cast<float>(value[2]) / 255.0f;
            float g = static_cast<float>(value[1]) / 255.0f;
            float b = static_cast<float>(value[0]) / 255.0f;
            input_vec.push_back(r);
            input_vec.push_back(g);
            input_vec.push_back(b);
        }
    }
    m_snpe_task->loadInputTensor(input_vec);
    return true;
}

bool
Detector::postprocess(cv::Mat &frame) {
    std::vector<float> output_vec;
    m_snpe_task->getOutputTensor(output_vec);

    float size[3]         = {13, 52, 26};
    float strides[3]      = {32,  8, 16};
    float anchorGrid[][6] = {
        {116, 90,156,198,373,326},  // 32 * 32
        { 10, 13, 16, 30, 33, 23},  //  8 * 8
        { 30, 61, 62, 45, 59,119},  // 16 * 16
    };

        
    for (size_t i = 0; i < 3; i++) {
        int height = size[i], width = size[i];
        for (int j = 0; j < height; j++) {      // 80/40/20
            for (int k = 0; k < width; k++) {   // 80/40/20
                int anchorIdx = 0;
                for (int l = 0; l < 3; l++) {   // 3
                    int index = 3 * (size[i] * j + k) + l;
                    if (i == 1) {
                        index += 507;
                    } else if (i == 2) {
                        index += 8619;
                    }
                    for (int m = 0; m < 5; m++) {
                        float value = 1.0 / (1.0 + exp(-static_cast<double>(output_vec[index * 28 + m])));
                        if (m < 2) {
                            float gridValue = m == 0 ? k : j;
                            output_vec[index * 28 + m] = (value * 2 - 0.5 + gridValue) * strides[i];
                        } else if (m < 4) {
                            output_vec[index * 28 + m] = value * value * 4 * anchorGrid[i][anchorIdx++];
                        } else {
                            output_vec[index * 28 + m] = value;
                        }
                    }
                }
            }
        }
    }

    std::vector<int>    box_idx;
    std::vector<float>  box_confidences;
    std::vector<Object> win_list;

    for (unsigned int i = 0; i < output_vec.size() / 28 /* grid size*/; i++) {
        float confidence = output_vec[i * 28 + 4];
        if (confidence > 0.3 /* confidence thres */) {
            box_idx.push_back(i);
            box_confidences.push_back(confidence);
        }
    }

    for (size_t i = 0; i < box_idx.size(); i++) {
        int idx = box_idx[i];
        float confidence = box_confidences[i];
        int max_idx = 5;
        for (int j = 6; j < 28; j++) {
            if (output_vec[idx * 28 + j] > output_vec[idx * 28 + max_idx]) max_idx = j;
        }

        float score = confidence * output_vec[idx * 28 + max_idx];
        if (score > 0.3 /* confidence thres */) {
            Object object;
            object.bbox.width  = output_vec[idx * 28 + 2];
            object.bbox.height = output_vec[idx * 28 + 3];
            object.bbox.x = std::max(0, static_cast<int>(output_vec[idx * 28] - object.bbox.width / 2));
            object.bbox.y = std::max(0, static_cast<int>(output_vec[idx * 28 + 1] - object.bbox.height / 2));

            object.label = max_idx - 5;
            object.confidence = confidence;
            win_list.push_back(object);
        }
    }

    win_list = nms(win_list, 0.5 /* nms thres */);
    for (auto &object: win_list) {
        cv::rectangle(frame, object.bbox, cv::Scalar(255, 255, 0), 2);
        std::string title = "Human " + std::to_string(int(object.confidence * 100)) + "%";
        cv::putText(
            frame, title, cv::Point(object.bbox.x, object.bbox.y - 5), 
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1, cv::LINE_AA
        );
    }
    cv::resize(frame, frame, cv::Size(1280, 720));
    return true;
}

std::vector<Object>
Detector::nms(std::vector<Object> win_list, const float &nms_thres) {
    if (win_list.empty()) return win_list;

    std::sort(win_list.begin(), win_list.end(), [] (const Object& left, const Object& right) {
        return (left.confidence > right.confidence);
    });

    std::vector<bool> flag(win_list.size(), false);
    for (unsigned int i = 0; i < win_list.size(); i++) {
        if (flag[i]) continue; 
        for (unsigned int j = i + 1; j < win_list.size(); j++) {
            if (calcIOU(win_list[i].bbox, win_list[j].bbox) > nms_thres) {
                flag[j] = true;
            }
        }
    }

    std::vector<Object> results;
    for (unsigned int i = 0; i < win_list.size(); i++) {
        if (!flag[i]) results.push_back(win_list[i]);
    }
    return results;
}

float 
Detector::calcIOU(const cv::Rect &a, const cv::Rect &b) {
    float x_overlap = std::max(
        0., std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x) + 1.
    );
    float y_overlap = std::max(
        0., std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y) + 1.
    );
    float intersection = x_overlap * y_overlap;
    float unio = (a.width + 1.) * (a.height + 1.) +
                 (b.width + 1.) * (b.height + 1.) - intersection;
    return intersection / unio;
}

}