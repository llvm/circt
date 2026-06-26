#ifndef HLS_INTERFACES_MEMORYDS_H
#define HLS_INTERFACES_MEMORYDS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"

namespace circt::hls_analysis {

    enum class StorageKind {
      RAM_1P,
      RAM_2P,
      RAM_S2P,
      RAM_T2P,
      LUTRAM,
      URAM,
      RAM_1WNR
    };

    struct PartitionSpec {
        enum Kind { Complete, Block, Cyclic } kind;
        unsigned factor;       // ignored for Complete
        unsigned dim;          // which dimension (0-indexed)
    };

    struct AccessSummary {
      mlir::Operation *op;
      bool isWrite;
      unsigned stageStart;
      unsigned latency;
      // Temporal grouping. Accesses with different `scope`s run in different
      // loops and do not contend for ports (unless in a dataflow region).
      mlir::Operation *scope = nullptr;
      // II of `scope` when it is a pipeline; 0 means sequential scope
      // (no cross-iteration overlap -> at most one access per cycle).
      unsigned ii = 0;
    };

    struct BRAMCost {
        StorageKind kind;
        unsigned bramCount;
        bool infeasible = false;   // true if port demand exceeds chosen kind
        std::string note;          // human-readable explanation
    };

    llvm::DenseMap<llvm::StringRef, PartitionSpec>
    getPartitionSpecs(mlir::Operation* op);

} // namespace circt::hls_analysis

#endif