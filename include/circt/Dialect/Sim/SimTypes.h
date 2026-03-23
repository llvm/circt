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

/// Return the keyword string for a DPIDirection (e.g. "input", "return").
llvm::StringRef stringifyDPIDirectionKeyword(DPIDirection dir);

/// Parse a keyword string to a DPIDirection. Returns std::nullopt on failure.
std::optional<DPIDirection> parseDPIDirectionKeyword(llvm::StringRef keyword);

/// True if a port with this direction is a call operand (input/inout/ref).
bool isCallOperandDir(DPIDirection dir);

/// A single port in a DPI module type.
struct DPIPort {
  mlir::StringAttr name;
  mlir::Type type;
  DPIDirection dir;
};

static inline bool operator==(const DPIPort &a, const DPIPort &b) {
  return a.dir == b.dir && a.name == b.name && a.type == b.type;
}
static inline llvm::hash_code hash_value(const DPIPort &port) {
  return llvm::hash_combine(static_cast<uint32_t>(port.dir), port.name,
                            port.type);
}

namespace detail {
struct DPIModuleTypeStorage : public mlir::TypeStorage {
  DPIModuleTypeStorage(llvm::ArrayRef<DPIPort> inPorts);

  using KeyTy = llvm::ArrayRef<DPIPort>;

  bool operator==(const KeyTy &key) const {
    return std::equal(key.begin(), key.end(), ports.begin(), ports.end());
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine_range(key.begin(), key.end());
  }

  static DPIModuleTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
    return new (allocator.allocate<DPIModuleTypeStorage>())
        DPIModuleTypeStorage(key);
  }

  KeyTy getAsKey() const { return ports; }

  llvm::ArrayRef<DPIPort> getPorts() const { return ports; }

  llvm::SmallVector<DPIPort> ports;
  llvm::SmallVector<size_t> inputToAbs;
  llvm::SmallVector<size_t> resultToAbs;
};
} // namespace detail

} // namespace sim
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Sim/SimTypes.h.inc"

#endif // CIRCT_DIALECT_SIM_SIMTYPES_H
