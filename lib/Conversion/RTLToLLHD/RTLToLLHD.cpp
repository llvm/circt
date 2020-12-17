//===- RTLToLLHD.cpp - RTL to LLHD Conversion Pass ------------------------===//
//
// This is the main RTL to LLHD Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/RTLToLLHD/RTLToLLHD.h"
#include "circt/Dialect/RTL/Ops.h"
#include "circt/Dialect/RTL/Visitors.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace llhd {
#define GEN_PASS_CLASSES
#include "circt/Conversion/RTLToLLHD/Passes.h.inc"
} // namespace llhd
} // namespace circt

using namespace circt;
using namespace llhd;

//===----------------------------------------------------------------------===//
// RTL to LLHD Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct RTLToLLHDPass
    : public ConvertRTLToLLHDBase<RTLToLLHDPass>,
      public rtl::CombinatorialVisitor<RTLToLLHDPass, LogicalResult>,
      public rtl::StmtVisitor<RTLToLLHDPass, LogicalResult> {
  void runOnOperation() override;
};
} // namespace

/// Create a RTL to LLHD conversion pass.
std::unique_ptr<OperationPass<rtl::RTLModuleOp>>
llhd::createConvertRTLToLLHDPass() {
  return std::make_unique<RTLToLLHDPass>();
}

/// Register the RTL to LLHD conversion pass.
namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Conversion/RTLToLLHD/Passes.h.inc"
} // namespace

void circt::llhd::registerRTLToLLHDPasses() { registerPasses(); }

/// This is the main entrypoint for the RTL to LLHD conversion pass.
void RTLToLLHDPass::runOnOperation() { getOperation().emitRemark("TODO"); }
