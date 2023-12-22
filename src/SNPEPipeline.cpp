#include "SNPEPipeline.hpp"

namespace snpe {

SNPEPipeline::SNPEPipeline() {
    m_snpe      = nullptr;
    m_container = nullptr;
}

bool
SNPEPipeline::init(const std::string &model_path) {
    static zdl::DlSystem::Version_t version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    std::cout << "[info] SNPE version: " << version.asString().c_str() << std::endl;

    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        m_runtim = zdl::DlSystem::Runtime_t::GPU;
        std::cout << "[info] SNPE runtim: GPU" << std::endl;
    } else {
        m_runtim = zdl::DlSystem::Runtime_t::CPU;
        std::cout << "[info] SNPE runtim: CPU" << std::endl;
    }

    zdl::DlSystem::StringList output_layers = {};
    const char *layers_name[3] = {"Conv_134", "Conv_148", "Conv_162"};
    for (auto &name: layers_name) {
        output_layers.append(name);
    }

    m_container = zdl::DlContainer::IDlContainer::open(model_path);
    if (m_container.get() == nullptr) {
        std::cerr << "[error] failed to load container." << std::endl;
        return false;
    }

    zdl::SNPE::SNPEBuilder snpe_builder(m_container.get());
    m_snpe = snpe_builder.setOutputLayers(output_layers)
                        .setRuntimeProcessor(m_runtim)
                        .setCPUFallbackMode(true)
                        .setUseUserSuppliedBuffers(false)
                        .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                        .build();
    return true;
}

void
SNPEPipeline::loadInputTensor(std::vector<float> &input_vec) {
    const auto &strList_opt = m_snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("failed to obtain input tensor name.");
    const auto &strList = *strList_opt;

    const auto &inputDims_opt = m_snpe->getInputDimensions(strList.at(0));
    const auto &input_shape = *inputDims_opt;
    m_input_tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(input_shape);
    std::copy(input_vec.begin(), input_vec.end(), m_input_tensor->begin());
}

void
SNPEPipeline::getOutputTensor(std::vector<float> &output_vec) {
    std::copy(m_output.begin(), m_output.end(), std::back_inserter(output_vec));
}

bool
SNPEPipeline::execute() {
    static zdl::DlSystem::TensorMap input_tmp, output_tmp;
    const auto strList = m_snpe->getInputTensorNames();
    input_tmp.add((*strList).at(0), m_input_tensor.get());

    m_snpe->execute(input_tmp, output_tmp);

    m_output.clear();
    zdl::DlSystem::StringList tensorNames = output_tmp.getTensorNames();
    std::for_each( tensorNames.begin(), tensorNames.end(), [&](const char* name) {
        auto tensorPtr = output_tmp.getTensor(name);
        for ( auto it = tensorPtr->cbegin(); it != tensorPtr->cend(); ++it) {
            m_output.push_back(*it);
        }
    });
    return true;
}

}