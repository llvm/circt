//===- PrintStateInfo.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_TRANSFORMS_PRINTSTATEINFO_H
#define CIRCT_DIALECT_ARC_TRANSFORMS_PRINTSTATEINFO_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace circt {
namespace arc {

/// Gathers information about a given Arc state.
struct StateInfo {
  enum Type { Input, Output, Register, Memory, Wire } type;
  std::string name;
  unsigned offset;
  unsigned numBits;
  unsigned memoryStride = 0; // byte separation between memory words
  unsigned memoryDepth = 0;  // number of words in a memory
};

/// Gathers information about a given Arc model.
struct ModelInfo {
  std::string name;
  size_t numStateBytes;
  std::vector<StateInfo> states;

  ModelInfo(std::string name, size_t numStateBytes,
            std::vector<StateInfo> states)
      : name(std::move(name)), numStateBytes(numStateBytes),
        states(std::move(states)) {}
};

/// Collects information about states within the provided Arc `storage`,
/// assuming default `offset`, and adds it to `stateInfos`.
mlir::LogicalResult collectStates(mlir::Value storage, unsigned offset,
                                  std::vector<StateInfo> &stateInfos);

/// Collects information about all Arc models in the provided `module`,
/// and adds it to `modelInfos`.
mlir::LogicalResult collectModels(mlir::ModuleOp module,
                                  std::vector<ModelInfo> &modelInfos);

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_TRANSFORMS_PRINTSTATEINFO_H
