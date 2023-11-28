#ifndef SNPE_PIPELINE_HPP_
#define SNPE_PIPELINE_HPP_

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlContainer/IDlContainer.hpp"

class SNPEPipeline {
public:
    SNPEPipeline();
    ~SNPEPipeline();

    bool init(const std::string &model_path);
    bool is_init() { return m_snpe != nullptr; }

    void loadInputTensor(std::vector<float> &input_vec);
    void getOutputTensor(std::vector<float> &output_vec);

    bool execute();

private:
    zdl::DlSystem::Runtime_t m_runtime;
    zdl::DlSystem::TensorMap m_input_map, m_output_map;
    
    std::unique_ptr<zdl::SNPE::SNPE> m_snpe;
    std::unique_ptr<zdl::DlContainer::IDlContainer> m_container;
};

#endif // SNPE_PIPELINE_HPP_