#ifndef HLS_PASSES_BRAMAUTOPIPELINE_H
#define HLS_PASSES_BRAMAUTOPIPELINE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt::hls_analysis {
std::unique_ptr<::mlir::Pass> createAutoPipelineUnrollPass();
void registerAutoPipelineUnrollPass();
} // namespace circt::hls_analysis

#endif