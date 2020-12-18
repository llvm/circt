//===- RTLToLLHD.h - LLHD to LLVM pass entry point --------------*- C++ -*-===//
//
// This header file defines prototypes that expose the RTLToLLHD pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_RTLTOLLHD_RTLTOLLHD_H_
#define CIRCT_CONVERSION_RTLTOLLHD_RTLTOLLHD_H_

#include "circt/Dialect/RTL/Ops.h"
#include <memory>

namespace mlir {
template <typename T>
class OperationPass;
class ModuleOp;
} // namespace mlir

namespace circt {
namespace llhd {
using namespace mlir;
std::unique_ptr<OperationPass<ModuleOp>> createConvertRTLToLLHDPass();
void registerRTLToLLHDPasses();
} // namespace llhd
} // namespace circt

#endif // CIRCT_CONVERSION_RTLTOLLHD_RTLTOLLHD_H_
