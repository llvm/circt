
#ifndef HLS_INTERFACES_MEMORYINTERFACE_H
#define HLS_INTERFACES_MEMORYINTERFACE_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/DialectInterface.h"
#include "circt/Dialect/Resource/Interfaces/MemoryDS.h"
#include "circt/Dialect/Resource/Interfaces/MemoryOpInterface.h.inc"
#include "circt/Dialect/Resource/Interfaces/MemoryDialectInterface.h.inc"

namespace circt::hls_analysis {
    // One-call registration that attaches everything to a DialectRegistry.
    void registerBRAMInterfaceExternalModels(mlir::DialectRegistry &registry);
} // namespace mlir::hls


#endif