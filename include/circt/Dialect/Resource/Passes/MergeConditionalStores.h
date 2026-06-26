#ifndef HLS_PASSES_ANALYSISPASSMERGECONDITIONAL_H
#define HLS_PASSES_ANALYSISPASSMERGECONDITIONAL_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt::hls_analysis {
std::unique_ptr<::mlir::Pass> createMergeConditionalStores();
void registerMergeConditionalStoresPass();
} // namespace circt::hls_analysis

#endif