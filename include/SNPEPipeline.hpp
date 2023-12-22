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

namespace snpe {

class SNPEPipeline {
public:
    SNPEPipeline();

    bool init(const std::string &model_path);
    bool isInit() { return m_snpe.get(); }

    void loadInputTensor(std::vector<float> &input_vec);
    void getOutputTensor(std::vector<float> &output_vec);

    bool execute();

private:
    std::vector<float> m_output;
    zdl::DlSystem::Runtime_t m_runtim;

    std::unique_ptr<zdl::SNPE::SNPE> m_snpe;
    std::unique_ptr<zdl::DlContainer::IDlContainer> m_container;
    std::unique_ptr<zdl::DlSystem::ITensor> m_input_tensor;
};

}


#endif // SNPE_PIPELINE_HPP_