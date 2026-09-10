#ifndef CIRCT_BRAM_PASSES
#define CIRCT_BRAM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def MemoryBanking : Pass<"memory-banking-bram", "::mlir::func::FuncOp"> {
  let summary = "Partition the memories used in kernel to banks";
  let constructor = "circt::hls_analysis::createMemoryBankingPassModified()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::affine::AffineDialect"
  ];
}

#endif // CIRCT_BRAM_PASSES