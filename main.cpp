#include <string>
#include <fstream>
#include <iostream>

#include "VideoPipeline.hpp"

int main(int argc, char **argv) {
    Json::Reader reader;
    std::ifstream input("/root/trip2-snpe/cfg/config.json");
    if (!input.is_open()) {
        std::cerr << "[error] failed to open configure files" << std::endl;
        return 1;
    }
    std::string contents((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
    input.close();

    Json::Value value;
    if (!reader.parse(contents, value)) {
        std::cerr << "[error] failed parsing json" << std::endl;
        return 1;
    }

    VideoPipeline *m_vp = new VideoPipeline(value);
    if (!m_vp->create("/root/trip2-snpe/yolov7_tiny.dlc")) {
        std::cerr << "[error] failed to create video pipeline instance." << std::endl;
        goto exit;
    }
    m_vp->start();

exit:
    if (m_vp) {
        delete m_vp;
        m_vp = nullptr;
    }
    return 0;
}