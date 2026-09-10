//===- BRAMClassification.h - Per-bank storage-mode analysis --------------===//
//
// A cached AnalysisManager analysis. For each on-chip bank (a memref produced
// by an alloca, after array-partition decomposition has run) it computes the
// worst-case concurrent access profile across every loop that touches the bank
// and classifies the required port mode: OnePort, SDP, or TDP.
//
// It does NOT rewrite the IR. The scheduler reads getMode(bank) to build typed
// resource limits; the BRAM estimator reads the profile (width/depth/mode) to
// pick a primitive. Nothing here is serialized.
//
//===----------------------------------------------------------------------===//

#ifndef BRAMOPT_ANALYSIS_BRAMCLASSIFICATION_H
#define BRAMOPT_ANALYSIS_BRAMCLASSIFICATION_H

#include "circt/Dialect/Resource/Interfaces/MemoryDS.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <circt/Dialect/Arc/ArcTypes.h.inc>

namespace circt::hls_analysis {

/// Physical port configuration a bank must support over its whole live range.
/// Ordered by capability: OnePort < SDP < TDP. The scheduler relies on this
/// ordering — defaulting to TDP never under-constrains, so tightening is the
/// only direction the post-schedule pass ever moves.
enum class StorageMode {
  OnePort, // 1 R/W port. Reads and writes contend.
  SDP,     // 1 dedicated read + 1 dedicated write.
  TDP,     // 2 ports, each R or W.
};

llvm::StringRef stringifyStorageMode(StorageMode mode);

/// Pure classification rule from a per-cycle access profile.
///   writes >= 2            -> TDP (two write-capable ports needed)
///   reads >= 1 && writes>=1-> SDP (1R + 1W dedicated)
///   reads >= 2             -> TDP (two read ports; see SDP-promotion TODO)
///   otherwise              -> OnePort
StorageMode classifyMode(unsigned reads, unsigned writes);

struct BankProfile {
  mlir::Value memref;          // The bank itself (alloca result).
  unsigned width = 0;          // Element bit width.
  unsigned depth = 0;          // Number of elements.
  unsigned maxReads = 0;       // Max concurrent reads/cycle across all loops.
  unsigned maxWrites = 0;      // Max concurrent writes/cycle across all loops.
  StorageMode mode = StorageMode::OnePort;
  StorageKind type =  StorageKind::RAM_1P;
  unsigned copies  = 1;                       // replication factor
  unsigned loopII  = 1;                       // binding II for this bank
};

class BRAMClassification {
public:
  /// Constructed by getAnalysis<BRAMClassification>(). `op` is the func (or any
  /// region-holding op) whose banks should be classified.
  explicit BRAMClassification(mlir::Operation *op);

  /// Mode for `bank`. Returns OnePort if the value is not a tracked bank.
  StorageMode getMode(mlir::Value bank) const;

  /// Full profile for `bank`, or nullptr if untracked.
  const BankProfile *getProfile(mlir::Value bank) const;

  const llvm::DenseMap<mlir::Value, BankProfile> &getBanks() const {
    return banks;
  }

  void dump(llvm::raw_ostream &os) const;

private:
  void enumerateBanks(mlir::Operation *root);
  void profileAccesses(mlir::Operation *root);
  void classify();

  llvm::DenseMap<mlir::Value, BankProfile> banks;
};

} // namespace bramopt

#endif // BRAMOPT_ANALYSIS_BRAMCLASSIFICATION_H
