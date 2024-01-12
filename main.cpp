#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "VideoPipeline.hpp"

class JsonReader {
public:
    JsonConfigReader(const std::string& filePath): input(filePath) {
        if (!input.is_open()) {
            throw std::runtime_error("Failed to open configure file");
        }
        contents = { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
    }

    Json::Value parseJson() {
        Json::Reader reader;
        Json::Value value;

        if (!reader.parse(contents, value)) {
            throw std::runtime_error("Failed parsing json");
        }
        return value;
    }

private:
    std::ifstream input;
    std::string contents;
};

int main(int argc, char **argv) {
    try {
        JsonReader configReader("/root/trip2-snpe/cfg/config.json");
        Json::Value value = configReader.parseJson();

        VideoPipeline m_vp(value);
        if (!m_vp.create(value["model"]["path"])) {
            std::cerr << "[error] failed to create video pipeline instance." << std::endl;
            return 1;
        }
        m_vp.start();
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
