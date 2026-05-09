//===- SimTypes.h - Sim dialect types ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMTYPES_H
#define CIRCT_DIALECT_SIM_SIMTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
namespace hw {
class ModuleType;
} // namespace hw
namespace sim {

// Forward-declare the tablegen enum (defined in SimEnums.h.inc).
enum class DPIDirection : uint32_t;

/// Return the keyword string for a DPIDirection (e.g. "in", "return").
llvm::StringRef stringifyDPIDirectionKeyword(DPIDirection dir);

/// Parse a keyword string to a DPIDirection. Returns std::nullopt on failure.
std::optional<DPIDirection> parseDPIDirectionKeyword(llvm::StringRef keyword);

/// True if an argument with this direction is a call operand (input/inout/ref).
bool isCallOperandDir(DPIDirection dir);

/// A single argument in a DPI function type.
struct DPIArgument {
  mlir::StringAttr name;
  mlir::Type type;
  DPIDirection dir;
};

static inline bool operator==(const DPIArgument &a, const DPIArgument &b) {
  return a.dir == b.dir && a.name == b.name && a.type == b.type;
}
static inline llvm::hash_code hash_value(const DPIArgument &arg) {
  return llvm::hash_combine(static_cast<uint32_t>(arg.dir), arg.name, arg.type);
}

namespace detail {
struct DPIFunctionTypeStorage : public mlir::TypeStorage {
  DPIFunctionTypeStorage(llvm::ArrayRef<DPIArgument> args);

  using KeyTy = llvm::ArrayRef<DPIArgument>;

  bool operator==(const KeyTy &key) const {
    return std::equal(key.begin(), key.end(), arguments.begin(),
                      arguments.end());
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine_range(key.begin(), key.end());
  }

  static DPIFunctionTypeStorage *
  construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key);

  KeyTy getAsKey() const { return arguments; }

  llvm::ArrayRef<DPIArgument> getArguments() const { return arguments; }

  /// Return the cached MLIR FunctionType for the call shape.
  mlir::FunctionType getCachedFunctionType() const { return cachedFuncType; }

  llvm::SmallVector<DPIArgument> arguments;
  llvm::SmallVector<size_t> inputToAbs;
  llvm::SmallVector<size_t> resultToAbs;
  mlir::FunctionType cachedFuncType;
};
} // namespace detail

} // namespace sim
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Sim/SimTypes.h.inc"

#endif // CIRCT_DIALECT_SIM_SIMTYPES_H
