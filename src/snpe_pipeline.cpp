#include "snpe_pipeline.hpp"

SNPEPipeline::SNPEPipeline() {
    m_snpe      = nullptr;
    m_container = nullptr;
}

SNPEPipeline::~SNPEPipeline() {

}

bool
SNPEPipeline::init(const std::string &model_path) {
    static zdl::DlSystem::Version_t version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    std::cout << "[info] snpe version: " << version.toString().c_str() << std::endl;

    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        m_runtime = zdl::DlSystem::Runtime_t::GPU;
        std::cout << "[info] snpe runtime: using GPU." << std::endl;
    } else {
        m_runtime = zdl::DlSystem::Runtime_t::CPU;
        std::cout << "[info] snpe runtime: using CPU." << std::endl; 
    }

    // create the dlc container
    m_container = zdl::DlContainer::IDlContainer::open(model_path);
    zdl::SNPE::SNPEBuilder snpeBuilder(m_container.get());
    m_snpe = snpeBuilder.setOutputLayers({})
                        .setRuntimeProcessor(m_runtime)
                        .setCPUFallbackMode(true)
                        .setUseUserSuppliedBuffers(false)
                        .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                        .build();
    return true;
}

void
SNPEPipeline::loadInputTensor(std::vector<float> &input_vec) {
    std::unique_ptr<zdl::DlSystem::ITensor> input_tensor;
    const auto &strList_opt = m_snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("error obtaining input tensor names");
    
    const auto &strList = *strList_opt;
    const auto &inputDims_opt = m_snpe->getInputDimensions(strList.at(0));
    const auto &input_shape = *inputDims_opt;
    input_tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(input_shape);
    std::copy(input_vec.begin(), input_vec.end(), input_tensor->begin());

    if (m_input_map.size() != 0) m_input_map.clear();
    m_input_map.add(strList.at(0), input_tensor.get());
}

void
SNPEPipeline::getOutputTensor(std::vector<float> &output_vec) {
    zdl::DlSystem::StringList tensor_names = m_output_map.getTensorNames();
    if (tensor_names.size() == 0) { return; }
    std::for_each(tensor_names.begin(), tensor_names.end(), [&](const char *name) {
        auto tensor_ptr = m_output_map.getTensor(name);
        for (auto it = tensor_ptr->cbegin(); it != tensor_ptr->cend(); ++it) {
            output_vec.push_back(*it);
        }
    });
}

bool
SNPEPipeline::execute() {
    if (m_output_map.size() != 0) m_output_map.clear();
    m_snpe->execute(m_input_map, m_output_map);
    return true;
}