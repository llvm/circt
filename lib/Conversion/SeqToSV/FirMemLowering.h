//===- FirMemLowering.h - FirMem lowering utilities ===========--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_SEQTOSV_FIRMEMLOWERING_H
#define CONVERSION_SEQTOSV_FIRMEMLOWERING_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"

namespace circt {

/// The configuration of a FIR memory.
struct FirMemConfig {
  size_t numReadPorts = 0;
  size_t numWritePorts = 0;
  size_t numReadWritePorts = 0;
  size_t dataWidth = 0;
  size_t depth = 0;
  size_t readLatency = 0;
  size_t writeLatency = 0;
  size_t maskBits = 0;
  seq::RUW readUnderWrite = seq::RUW::Undefined;
  seq::WUW writeUnderWrite = seq::WUW::Undefined;
  SmallVector<int32_t, 1> writeClockIDs;
  StringRef initFilename;
  bool initIsBinary = false;
  bool initIsInline = false;
  Attribute outputFile;
  StringRef prefix;

  llvm::hash_code hashValue() const {
    return llvm::hash_combine(numReadPorts, numWritePorts, numReadWritePorts,
                              dataWidth, depth, readLatency, writeLatency,
                              maskBits, readUnderWrite, writeUnderWrite,
                              initFilename, initIsBinary, initIsInline,
                              outputFile, prefix) ^
           llvm::hash_combine_range(writeClockIDs.begin(), writeClockIDs.end());
  }

  auto getTuple() const {
    return std::make_tuple(numReadPorts, numWritePorts, numReadWritePorts,
                           dataWidth, depth, readLatency, writeLatency,
                           maskBits, readUnderWrite, writeUnderWrite,
                           writeClockIDs, initFilename, initIsBinary,
                           initIsInline, outputFile, prefix);
  }

  bool operator==(const FirMemConfig &other) const {
    return getTuple() == other.getTuple();
  }
};

/**
 * FIR memory lowering helper.
 */
class FirMemLowering {
public:
  /// A vector of unique `FirMemConfig`s and all the `FirMemOp`s that use it.
  using UniqueConfigs =
      llvm::MapVector<FirMemConfig, SmallVector<seq::FirMemOp, 1>>;

  /// Information required to lower a single memory in a module.
  using MemoryConfig =
      std::tuple<FirMemConfig *, hw::HWModuleGeneratedOp, seq::FirMemOp>;

  FirMemLowering(ModuleOp circuit);

  /**
   * Groups memories by their kind from the whole design.
   */
  UniqueConfigs collectMemories(ArrayRef<hw::HWModuleOp> modules);

  /**
   * Lowers a group of memories from the same module.
   */
  void lowerMemoriesInModule(hw::HWModuleOp module,
                             ArrayRef<MemoryConfig> mems);

  /**
   * Creates the generated module for a given configuration.
   */
  hw::HWModuleGeneratedOp createMemoryModule(FirMemConfig &mem,
                                             ArrayRef<seq::FirMemOp> memOps);

private:
  FirMemConfig collectMemory(seq::FirMemOp op);

  /**
   * Find the schema or create it if it does not exist.
   */
  FlatSymbolRefAttr getOrCreateSchema();

private:
  MLIRContext *context;
  ModuleOp circuit;

  SymbolCache symbolCache;
  Namespace globalNamespace;

  DenseMap<hw::HWModuleOp, size_t> moduleIndex;

  hw::HWGeneratorSchemaOp schemaOp;
};

} // namespace circt

namespace llvm {
template <>
struct DenseMapInfo<circt::FirMemConfig> {
  static inline circt::FirMemConfig getEmptyKey() {
    circt::FirMemConfig cfg;
    cfg.depth = DenseMapInfo<size_t>::getEmptyKey();
    return cfg;
  }
  static inline circt::FirMemConfig getTombstoneKey() {
    circt::FirMemConfig cfg;
    cfg.depth = DenseMapInfo<size_t>::getTombstoneKey();
    return cfg;
  }
  static unsigned getHashValue(const circt::FirMemConfig &cfg) {
    return cfg.hashValue();
  }
  static bool isEqual(const circt::FirMemConfig &lhs,
                      const circt::FirMemConfig &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // CONVERSION_SEQTOSV_FIRMEMLOWERING_H
