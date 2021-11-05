#include "circt/Dialect/HIR/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

void circt::hir::registerHIRLoweringPassPipeline() {
  // Register an inline pipeline builder.
  mlir::PassPipelineRegistration<>(
      "hir-simplify",
      "Simplify HIR dialect to a bare minimum for lowering to verilog.",
      [](mlir::OpPassManager &pm) {
        // -canonicalize -hir-loop-unroll -canonicalize -sccp
        // -hir-lower-memref -sccp -cse -canonicalize
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(circt::hir::createLoopUnrollPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createSCCPPass());
        pm.addPass(circt::hir::createMemrefLoweringPass());
        pm.addPass(hir::createSimplifyLoopPass());
        pm.addPass(mlir::createSCCPPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());
      });
}
