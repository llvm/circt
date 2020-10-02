//===- FIRRTLToLLHD.cpp - FIRRTL to LLHD Conversion Pass ------------------===//
//
// This is the main FIRRTL to LLHD Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FIRRTLToLLHD/FIRRTLToLLHD.h"

#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace llhd {
#define GEN_PASS_CLASSES
#include "circt/Conversion/FIRRTLToLLHD/Passes.h.inc"
}  // namespace llhd
}  // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::llhd;

namespace {
struct FIRRTLToLLHDConversionPass
    : public ConvertFIRRTLToLLHDBase<FIRRTLToLLHDConversionPass> {
  void runOnOperation() override {}
};
}  // namespace

/// Create an FIRRTL to LLHD conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> llhd::createConvertFIRRTLToLLHDPass() {
  return std::make_unique<FIRRTLToLLHDConversionPass>();
}

/// Register the FIRRTL to LLHD convesion pass.
namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Conversion/FIRRTLToLLHD/Passes.h.inc"
}  // namespace

void llhd::registerFIRRTLToLLHDPasses() { registerPasses(); }
