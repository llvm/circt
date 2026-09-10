#ifndef MLIR_INTERFACES_DIALECTMEMORYINTERFACE
#define MLIR_INTERFACES_DIALECTMEMORYINTERFACE

include "mlir/IR/Interfaces.td"

def MemoryResourceOpInterface : OpInterface<"MemoryResourceOpInterface"> {
  let description = [{
    Per-op interface for allocations whose memory cost should be estimated.
  }];
  
  let cppNamespace = "circt::hls_analysis";

    let methods = [
      InterfaceMethod<"Element type of the underlying storage.",
                      "::mlir::Type", "getElementType">,
      InterfaceMethod<"Static shape; empty if dynamic.",
                      "::llvm::SmallVector<int64_t>", "getStaticShape">,
      InterfaceMethod<"Total static size in bits.",
                      "int64_t", "getStaticSizeInBits">,
      InterfaceMethod<"Pragma-requested storage kind, if any.",
                      "::std::optional<::circt::hls_analysis::StorageKind>", "getPragmaStorageKind">,
      InterfaceMethod<"Partitioning spec from array_partition pragma.",
                      "::std::optional<::circt::hls_analysis::PartitionSpec>", "getPartitionSpec">,
      InterfaceMethod<"True if this op participates in cost accounting.",
                      "bool", "contributesToBRAMCost">,
    ];
}

#endif
