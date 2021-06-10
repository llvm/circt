#include "circt/Transforms/HIR/Passes.h"
#include "mlir/Pass/Pass.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Transforms/HIR/Passes.h.inc"
} // namespace

void mlir::hir::initHIRTransformationPasses() {
  registerPasses();
  registerHIRLoweringPassPipeline();
}
