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

#ifndef CIRCT_DIALECT_ARC_MODELINFO_H
#define CIRCT_DIALECT_ARC_MODELINFO_H

#include "circt/Dialect/Arc/ArcOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

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
  llvm::SmallVector<StateInfo> states;
  mlir::FlatSymbolRefAttr initialFnSym;
  mlir::FlatSymbolRefAttr finalFnSym;

  ModelInfo(std::string name, size_t numStateBytes,
            llvm::SmallVector<StateInfo> states,
            mlir::FlatSymbolRefAttr initialFnSym,
            mlir::FlatSymbolRefAttr finalFnSym)
      : name(std::move(name)), numStateBytes(numStateBytes),
        states(std::move(states)), initialFnSym(initialFnSym),
        finalFnSym(finalFnSym) {}
};

struct ModelInfoAnalysis {
  explicit ModelInfoAnalysis(Operation *container);
  llvm::MapVector<ModelOp, ModelInfo> infoMap;
};

/// Collects information about states within the provided Arc model storage
/// `storage`,  assuming default `offset`, and adds it to `states`.
mlir::LogicalResult collectStates(mlir::Value storage, unsigned offset,
                                  llvm::SmallVector<StateInfo> &states);

/// Collects information about all Arc models in the provided `module`,
/// and adds it to `models`.
mlir::LogicalResult collectModels(mlir::ModuleOp module,
                                  llvm::SmallVector<ModelInfo> &models);

/// Serializes `models` to `outputStream` in JSON format.
void serializeModelInfoToJson(llvm::raw_ostream &outputStream,
                              llvm::ArrayRef<ModelInfo> models);

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_MODELINFO_H
