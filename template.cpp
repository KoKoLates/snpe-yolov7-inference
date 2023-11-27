
#include <queue>
#include <mutex>
#include <thread>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlContainer/IDlContainer.hpp"

std::mutex queueMutex;
std::queue<cv::Mat> frameQueue;
std::unique_ptr<zdl::SNPE::SNPE> snpe;

struct BBox {
    cv::Rect bbox;
    int label = -1;
    float confidence = -1.0f;
};

static float calcIoU(const cv::Rect& a, const cv::Rect& b) {
    float xOverlap = std::max(
        0.,
        std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x) + 1.);
    float yOverlap = std::max(
        0.,
        std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y) + 1.);
    float intersection = xOverlap * yOverlap;
    float unio =
        (a.width + 1.) * (a.height + 1.) +
        (b.width + 1.) * (b.height + 1.) - intersection;
    return intersection / unio;
}

std::vector<BBox> nms(std::vector<BBox> winList, const float& nms_thresh) {
    if (winList.empty()) {
        return winList;
    }

    std::sort(winList.begin(), winList.end(), [] (const BBox& left, const BBox& right) {
        if (left.confidence > right.confidence) {
            return true;
        } else {
            return false;
        }
    });

    std::vector<bool> flag(winList.size(), false);
    for (unsigned int i = 0; i < winList.size(); i++) {
        if (flag[i]) {
            continue;
        }

        for (unsigned int j = i + 1; j < winList.size(); j++) {
            if (calcIoU(winList[i].bbox, winList[j].bbox) > nms_thresh) {
                flag[j] = true;
            }
         }
    }

    std::vector<BBox> ret;
    for (unsigned int i = 0; i < winList.size(); i++) {
        if (!flag[i])
            ret.push_back(winList[i]);
    }

    return ret;
}

zdl::DlSystem::Runtime_t checkRuntime()
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    std::cout << "[info] snpe version: " << Version.asString().c_str() << std::endl;
    static zdl::DlSystem::Runtime_t Runtime;
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        Runtime = zdl::DlSystem::Runtime_t::GPU;
        std::cout << "[info] runtime usage: GPU." << std::endl;
    } else {
        Runtime = zdl::DlSystem::Runtime_t::CPU;
        std::cout << "[info] runtime usage: CPU." <<std::endl;
    }
    return Runtime;
}

std::unique_ptr<zdl::SNPE::SNPE> initializeSNPE(zdl::DlSystem::Runtime_t runtime) {
    zdl::DlSystem::StringList outputLayers = {};
    outputLayers.append("Conv_134");
    outputLayers.append("Conv_148");
    outputLayers.append("Conv_162");
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open("/home/root/voxl_test/yolo_new.dlc");   
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    std::unique_ptr<zdl::SNPE::SNPE> snpe = snpeBuilder.setOutputLayers(outputLayers)
                      .setRuntimeProcessor(runtime)
                      .setCPUFallbackMode(true)
                      .setUseUserSuppliedBuffers(false)
                      .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                      .build();                 
    return snpe;
}

std::unique_ptr<zdl::DlSystem::ITensor>
loadInputTensor(std::unique_ptr<zdl::SNPE::SNPE> &snpe, std::vector<float> inputVec) {
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("error obtaining Input tensor names");
    const auto &strList = *strList_opt;

    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;
    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

    std::copy(inputVec.begin(), inputVec.end(), input->begin());
    return input;
}

std::vector<float> 
executeNetwork(std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                zdl::DlSystem::TensorMap inputTensorMap) {
    static zdl::DlSystem::TensorMap outputTensorMap;
    snpe->execute(inputTensorMap, outputTensorMap);
    std::vector<float> output;
    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
    std::for_each( tensorNames.begin(), tensorNames.end(), [&](const char* name)
    {
        auto tensorPtr = outputTensorMap.getTensor(name);
        for ( auto it = tensorPtr->cbegin(); it != tensorPtr->cend(); ++it )
        {
            output.push_back(*it);
        }
    });
    return output;
}


void captureFrames(cv::VideoCapture& cap) {
    cv::Mat frame;
    while (true) {
        cap.read(frame);
        if (frame.empty()) {
            break;
        }

        // Lock the queue before pushing a frame
        std::lock_guard<std::mutex> lock(queueMutex);
        frameQueue.push(frame);
    }
}

void processAndWriteFrames(cv::VideoWriter& out) {
    while (true) {
        cv::Mat image;

        // Lock the queue before popping a frame
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (!frameQueue.empty()) {
                image = frameQueue.front();
                frameQueue.pop();
            }
        }

        if (image.empty()) {
            // Sleep for a short time if the queue is empty
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // process here
        float down_width = 640, down_height = 640;
        resize(image, image, cv::Size(640, 640), cv::INTER_LINEAR);

        //push_input_vector
        std::vector<float> pixel;
        for (float y = 0; y < down_width; y++) {
            for (float x = 0; x < down_height; x++) {
                cv::Vec3b pixel_value = image.at<cv::Vec3b>(y, x);
                float red   = static_cast<float>(pixel_value[2]) / 255.0f;
                float green = static_cast<float>(pixel_value[1]) / 255.0f;
                float blue  = static_cast<float>(pixel_value[0]) / 255.0f;
                pixel.push_back(red);
                pixel.push_back(green);
                pixel.push_back(blue);
            }
        }
        
        std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, pixel);
        static zdl::DlSystem::TensorMap inputTensorMap;
        const auto strList = snpe->getInputTensorNames();
        inputTensorMap.add((*strList).at(0), inputTensor.get());

        std::vector<float> output = executeNetwork(snpe, inputTensorMap);

        float strides[3] = {32, 8, 16};
        float anchorGrid[][6] = {
            {142,110, 192,243, 459,401},  // 32*32
            {12,16, 19,36, 40,28},       // 8*8
            {36,75, 76,55, 72,146},      // 16*16
        };

        float size[3]={20,80,40};
        for (size_t i = 0; i < 3; i++) {
            int height = size[i];
            int width = size[i];
            for (int j = 0; j < height; j++) {      // 80/40/20
                for (int k = 0; k < width; k++) {   // 80/40/20
                    int anchorIdx = 0;
                    for (int l = 0; l < 3; l++) {   // 3
                        int index = 3*(size[i]*j+k)+l;
                        if (i == 1) {
                            index += 1200;
                        } else if (i == 2) {
                            index += 20400;
                        }
                        for (int m = 0; m < 4; m++) {     // 85
                            float value = 1.0/(1.0+exp(-static_cast<double>(output[index * 85 + m])));
                            if (m < 2) {
                                float gridValue = m == 0 ? k : j;
                                output[index * 85 + m] = (value * 2 - 0.5 + gridValue) * strides[i];
                            } else if (m < 4) {
                                output[index * 85 + m] = value * value * 4 * anchorGrid[i][anchorIdx++];
                            } 
                        }
                    }
                }
            }
        }

        std::vector<int> box_index;
        std::vector<float> box_confidences;
        std::vector<BBox> winList;

        for(int i = 0; i < 1200; i++) {
            float confidence = output[i * 85 + 4];
            if (confidence > 0.3) {
                box_index.push_back(i);
                box_confidences.push_back(confidence);
            }
        }

        for (size_t i = 0; i < box_index.size(); i++) {
            int current_idx = box_index[i];
            float current_cfd = box_confidences[i];
            int max_idx = 5;
            for (int j = 6; j < 85; j++) {
                if (output[current_idx * 85 + j] > output[current_idx * 85 + max_idx]) max_idx = j;
            }

            float score = current_cfd * output[current_idx * 85 + max_idx];
            if (score > 0.3) {
                BBox rect;
                rect.bbox.width = output[current_idx * 85 + 2];
                rect.bbox.height = output[current_idx * 85 + 3];
                rect.bbox.x = std::max(0, static_cast<int>(output[current_idx * 85] - rect.bbox.width / 2));
                rect.bbox.y = std::max(0, static_cast<int>(output[current_idx * 85 + 1] - rect.bbox.height / 2));

                rect.label = max_idx - 5;
                rect.confidence = current_cfd;
                winList.push_back(rect);
            }
        }

        winList = nms(winList, 0.5);

        for (auto& rect: winList) {
            if (rect.label == 0) {
                cv::rectangle(image, rect.bbox, cv::Scalar(255, 0, 0), 2);
            }
        }

        cv::resize(image, image, cv::Size(640, 480));
        out.write(image);
    }
}


int main(int argc, char **argv) {
    // video streaming init
    cv::VideoCapture cap("rtsp://localhost:8900/live");

    if (!cap.isOpened()) {
        std::cout << "Video capture failed to load." << std::endl;
        return -1;
    }

    cv::VideoWriter out(
        "appsrc ! videoconvert ! video/x-raw, format=I420 ! x264enc speed-preset=superfast tune=zerolatency ! "
        "h264parse config-interval=1 ! rtph264pay ! udpsink host=192.168.50.179 port=11024 buffer-size=524288",
        cv::CAP_GSTREAMER, 30, cv::Size(640, 480)  // Adjusted resolution
    );

    if (!out.isOpened()) {
        std::cout << "Video writer failed to load." << std::endl;
        return -1;
    }
    std::cout << "[info] Initialization successful." << std::endl;

    // snpe init
    zdl::DlSystem::Runtime_t runtime = checkRuntime();
    snpe = initializeSNPE(runtime);

    // Create threads for capturing and processing/writing frames
    std::thread captureThread(captureFrames, std::ref(cap));
    std::thread processThread(processAndWriteFrames, std::ref(out));

    // Wait for threads to finish
    captureThread.join();
    processThread.join();

    cap.release();
    out.release();

    return 0;
}
