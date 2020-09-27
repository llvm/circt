//===- PassRegistration.cpp - Register LLHD transformation passes ---------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"
} // namespace

void circt::llhd::initLLHDTransformationPasses() { registerPasses(); }
