//===- ModelInfo.h - Information about Arc models -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines and computes information about Arc models.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_ARCILATOR_MODELINFO
#define CIRCT_TOOLS_ARCILATOR_MODELINFO

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace circt {
namespace arcilator {

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

/// Collects information about states within the provided Arc model storage
/// `storage`,  assuming default `offset`, and adds it to `states`.
mlir::LogicalResult collectStates(mlir::Value storage, unsigned offset,
                                  std::vector<StateInfo> &states);

/// Collects information about all Arc models in the provided `module`,
/// and adds it to `models`.
mlir::LogicalResult collectModels(mlir::ModuleOp module,
                                  std::vector<ModelInfo> &models);

/// Serializes `models` to `outputStream` in JSON format.
void serializeModelInfoToJson(llvm::raw_ostream &outputStream,
                              std::vector<ModelInfo> &models);

} // namespace arcilator
} // namespace circt

#endif // CIRCT_TOOLS_ARCILATOR_MODELINFO