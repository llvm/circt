#ifndef HLS_PASSES_MEMORYBANKINGMODIFIED_H
#define HLS_PASSES_MEMORYBANKINGMODIFIED_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt::hls_analysis {

static std::unique_ptr<mlir::Pass>
createMemoryBankingPassModified();

void registerMemoryBankingPassModified();
} // namespace circt::hls_analysis

#endif