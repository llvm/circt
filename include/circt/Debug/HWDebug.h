#ifndef CIRCT_HWDEBUG_H_
#define CIRCT_HWDEBUG_H_

#include "mlir/Pass/Pass.h"

namespace circt::debug {
std::unique_ptr<mlir::Pass> createExportHGDBPass(std::string filename);
}

#endif // CIRCT_HWDEBUG_H_
