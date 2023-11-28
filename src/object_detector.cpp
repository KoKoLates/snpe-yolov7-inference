#include "object_detector.hpp"

namespace yolov7 {

Detector::Detector() {

}

Detector::~Detector() {
    
}

bool
Detector::init(const std::string &model_path) {
    snpe_task = std::move(std::unique_ptr<SNPEPipeline>(new SNPEPipeline()));
    if (!snpe_task->init(model_path)) {
        std::cerr << "[error] yolov7 detector: failed to initialize snpe instance." << std::endl;
        return false;
    }
    m_init = true;
    return false;
}

bool
Detector::detect(cv::Mat &image, std::vector<BBox> &results) {
    if (!preprocess(image)) {
        std::cerr << "[error] preprocess failed" << std::endl;
        return false;
    }

    if (!snpe_task->execute()) {
        std::cerr << "[error] snpe inference failed" << std::endl;
        return false;
    }

    if (!postprocess(image, results)) {
        std::cerr << "[error] postprocess failed" << std::endl;
        return false;
    }
    return true;
}

bool
Detector::preprocess(cv::Mat &image) {
    int target_w = 416, target_h = 416;
    cv::resize(image, image, cv::Size(target_w, target_h), cv::INTER_LINEAR);

    std::vector<float> input_vec;
    for (int y = 0; y < target_w; y++) {
        for (int x = 0; x < target_h; x++) {
            cv::Vec3b value = image.at<cv::Vec3b>(y, x);
            float r = static_cast<float>(value[2]) / 255.f;
            float g = static_cast<float>(value[1]) / 255.f;
            float b = static_cast<float>(value[0]) / 255.f;
            input_vec.push_back(r);
            input_vec.push_back(g);
            input_vec.push_back(b);
        }
    }
    snpe_task->loadInputTensor(input_vec);
}

bool
Detector::postprocess(cv::Mat &image, std::vector<BBox> &results) {
    std::vector<float> output_vec;
    snpe_task->getOutputTensor(output_vec);

    float grids[3]   = {13, 52, 26};
    float strides[3] = {32, 8, 16};
    float anchorGrid[][6] = {
        {116, 90,156,198,373,326},  // 32 * 32
        { 10, 13, 16, 30, 33, 23},  //  8 * 8
        { 30, 61, 62, 45, 59,119},  // 16 * 16
    };

    for (size_t i = 0; i < 3; i++) {
        int height = grids[i], width = grids[i];
        for (int j = 0; j < height; j++) {      // 80/40/20
            for (int k = 0; k < width; k++) {   // 80/40/20
                int anchorIdx = 0;
                for (int l = 0; l < 3; l++) {   // 3
                    int index = 3 * (grids[i] * j + k) + l;
                    if (i == 1) {
                        index += 507;
                    } else if (i == 2) {
                        index += 8619;
                    }
                    for (int m = 0; m < 5; m++) {     // 85
                        float value = 1.0 / (1.0+exp(-static_cast<double>(output_vec[index * 28 + m])));
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

    std::vector<BBox>   win_list;
    std::vector<int>    box_index;
    std::vector<float>  box_confidence;
    
    for (int i = 0; i < output_vec.size() / 28; i++) {
        float confidence = output_vec[i * 28 + 4];
        if (confidence > 0.3 /* confidence thres */) {
            box_index.push_back(i);
            box_confidence.push_back(confidence);
        }
    }

    for (size_t i = 0; i < box_index.size(); i++) {
        int current_idx = box_index[i];
        float current_cfd = box_confidence[i];
        int max_idx = 5;
        for (int j = 6; j < 28; j++) {
            if (output_vec[current_idx * 28 + j] > output_vec[current_idx * 28 + max_idx]) max_idx = j;
        }

        float score = current_cfd * output_vec[current_idx * 28 + max_idx];
        if (score > 0.3) {
            BBox rect;
            rect.bbox.width = output_vec[current_idx * 28 + 2];
            rect.bbox.height = output_vec[current_idx * 28 + 3];
            rect.bbox.x = std::max(0, static_cast<int>(output_vec[current_idx * 28] - rect.bbox.width / 2));
            rect.bbox.y = std::max(0, static_cast<int>(output_vec[current_idx * 28 + 1] - rect.bbox.height / 2));

            rect.label = max_idx - 5;
            rect.confidence = current_cfd;
            win_list.push_back(rect);
        }
    }

    win_list = nms(win_list, 0.5);
    for (auto &rect: win_list) {
        if (rect.label == 0) {
            cv::rectangle(image, rect.bbox, cv::Scalar(255, 0, 0), 2);
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << rect.confidence;
            std::string rounded_string = stream.str();
            std::string class_name = "human" + rounded_string;
            cv::putText(image, class_name, cv::Point(rect.bbox.x, rect.bbox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.3 , cv::Scalar(255, 0, 0), 1 , cv::LINE_AA);
        }
    }
}

std::vector<BBox>
Detector::nms(std::vector<BBox> win_list, const float &nms_thres) {
    if (win_list.empty()) {
        return win_list;
    }

    std::sort(win_list.begin(), win_list.end(), [] (const BBox& left, const BBox& right) {
        if (left.confidence > right.confidence) {
            return true;
        } else {
            return false;
        }
    });

    std::vector<bool> flag(win_list.size(), false);
    for (int i = 0; i < win_list.size(); i++) {
        if (flag[i]) {
            continue;
        }

        for (int j = i + 1; j < win_list.size(); j++) {
            if (calculate_iou(win_list[i].bbox, win_list[j].bbox) > nms_thres) {
                flag[j] = true;
            }
        }
    }

    std::vector<BBox> ret;
    for (int i = 0; i < win_list.size(); i++) {
        if (!flag[i])
            ret.push_back(win_list[i]);
    }
    return ret;

}

float Detector::calculate_iou(const cv::Rect &a, const cv::Rect &b) {
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