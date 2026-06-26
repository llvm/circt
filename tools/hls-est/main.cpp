#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "circt/Dialect/Resource/Interfaces/MemoryInterface.h"
#include "circt/Dialect/Resource/Passes/AffineToLoopScheduleModified.h"
#include "circt/Dialect/Resource/Passes/BRAMAnalysis.h"
#include "circt/Dialect/Resource/Passes/MergeConditionalStores.h"
#include "circt/Dialect/Resource/Passes/MemoryBankModified.h"
#include "circt/Dialect/Resource/Passes/AutoPipelineUnroll.h"
#include "circt/Dialect/Resource/Passes/DebugUnroll.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"   // createCSEPass, registerCSEPass, registerCanonicalizerPass et al.

#include "mlir/Dialect/Affine/IR/AffineOps.h"          // affine::AffineDialect
#include "mlir/Dialect/Arith/IR/Arith.h"               // arith::ArithDialect
#include "mlir/Dialect/MemRef/IR/MemRef.h"             // memref::MemRefDialect
#include "mlir/Dialect/SCF/IR/SCF.h"                   // scf::SCFDialect
#include "mlir/Dialect/Func/IR/FuncOps.h"              // func::FuncDialect
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"   // cf::ControlFlowDialect
#include "mlir/Dialect/Index/IR/IndexDialect.h"        // index::IndexDialect
#include "mlir/Dialect/Affine/Transforms/Passes.h"     // affine-loop-normalize et al.

// CIRCT dialects
#include "circt/Dialect/HW/HWDialect.h"                // hw::HWDialect
#include "circt/Dialect/Comb/CombDialect.h"            // comb::CombDialect
#include "circt/Dialect/Seq/SeqDialect.h"              // seq::SeqDialect
#include "circt/Dialect/Calyx/CalyxDialect.h"          // calyx::CalyxDialect
#include "circt/Dialect/Resource/HLS/HLSOps.h"         // custom dialect

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<
      // Upstream
      mlir::affine::AffineDialect,
      mlir::arith::ArithDialect,
      mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect,
      mlir::func::FuncDialect,
      mlir::cf::ControlFlowDialect,
      mlir::index::IndexDialect,
      // CIRCT
      circt::loopschedule::LoopScheduleDialect,
      circt::hw::HWDialect,
      circt::comb::CombDialect,
      circt::seq::SeqDialect,
      circt::calyx::CalyxDialect,
      circt::hls_analysis::HLSDialect
  >();

  // External interfaces still need their host dialects loaded above.
  circt::hls_analysis::registerBRAMInterfaceExternalModels(registry);
  circt::hls_analysis::registerBRAMLoopscheduleAnalysisPass();
  circt::hls_analysis::registerAffineToLoopScheduleAnalysisPass();
  circt::hls_analysis::registerMergeConditionalStoresPass();
  circt::hls_analysis::registerMemoryBankingPassModified();
  circt::hls_analysis::registerAutoPipelineUnrollPass();
  circt::hls_analysis::registerDebugUnrollPass();
  // Upstream affine passes (gives you -affine-loop-normalize, -affine-loop-unroll,
  // -affine-loop-coalescing, etc. on the command line).
  mlir::affine::registerAffinePasses();
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HLS BRAM analysis driver\n", registry));
}