#ifndef MLIR_INTERFACES_DIALECTINLINERINTERFACE
#define MLIR_INTERFACES_DIALECTINLINERINTERFACE

include "mlir/IR/Interfaces.td"

def MemoryResourceOpInterface : OpInterface<"MemoryResourceOpInterface"> {
  let description = [{
    Per-op interface for allocations whose BRAM cost should be estimated.
  }];
  
  let cppNamespace = "hls";

    let methods = [
      InterfaceMethod<"Element type of the underlying storage.",
                      "::mlir::Type", "getElementType">,
      InterfaceMethod<"Static shape; empty if dynamic.",
                      "::llvm::SmallVector<int64_t>", "getStaticShape">,
      InterfaceMethod<"Total static size in bits.",
                      "int64_t", "getStaticSizeInBits">,
      InterfaceMethod<"Pragma-requested storage kind, if any.",
                      "::std::optional<::hls::StorageKind>", "getPragmaStorageKind">,
      InterfaceMethod<"Partitioning spec from array_partition pragma.",
                      "::std::optional<::hls::PartitionSpec>", "getPartitionSpec">,
      InterfaceMethod<"Pipeline II of the enclosing loop, if any.",
                      "::std::optional<int64_t>", "getEnclosingPipelineII">,
      // The new one: enumerate accesses with timing.
      InterfaceMethod<"All loads/stores to this allocation, with stage info.",
                      "::llvm::SmallVector<::hls::AccessSummary>", "getAccessSummary">,
      InterfaceMethod<"True if this op participates in cost accounting.",
                      "bool", "contributesToBRAMCost">,
    ];
}

def BRAMEstimationInterface : DialectInterface<"BRAMEstimationInterface"> {
  let description = [{
    Dialect-level cost policies: port overhead, inference overrides,
    op filtering.
  }];
  
  let cppNamespace = "hls";

  let methods = [
    InterfaceMethod<[{
      True if this op should be skipped entirely by the analysis.
    }], "void", "shouldSkip">,

    InterfaceMethod<[{
      BRAM cost from interface ports owned by this dialect (e.g. M_AXI
      adds 2 BRAMs per port on Xilinx).
    }], "void", "getInterfacePortCost">,

    InterfaceMethod<[{
      Dialect-specific override of storage-kind resolution. If this
      returns a value, the OpInterface default is bypassed.
    }], "void", "overrideStorageKind">,

    InterfaceMethod<[{
      Hook for dialect-specific cost adjustments after per-op estimate
      (e.g. dataflow-region double-buffering multiplier).
    }], "void", "adjustCost">,
  ];
}

#endif
