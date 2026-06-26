#ifndef HLS_PASSES_BRAMANALYSISPASSLOOPSCHEDULE_H
#define HLS_PASSES_BRAMANALYSISPASSLOOPSCHEDULE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt::hls_analysis {
    std::unique_ptr<::mlir::Pass> createBRAMAnalysisPass();
    void registerBRAMLoopscheduleAnalysisPass();
} // namespace circt::hls_analysis

#endif