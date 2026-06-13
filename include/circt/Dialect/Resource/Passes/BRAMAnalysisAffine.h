#ifndef HLS_PASSES_BRAMANALYSISPASSAFFINE_H
#define HLS_PASSES_BRAMANALYSISPASSAFFINE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace hls {
std::unique_ptr<::mlir::Pass> createBRAMAffineAnalysis();
void registerBRAMAffineAnalysisPass();
} // namespace hls

#endif