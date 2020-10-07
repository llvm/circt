//===- Types.h - types for the RTL dialect ----------------------*- C++ -*-===//
//
// Types for the RTL dialect are mostly in tablegen. This file should contain
// C++ types used in MLIR type parameters.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESITYPES_HPP
#define CIRCT_DIALECT_ESI_ESITYPES_HPP

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "Dialect.h"

namespace circt {
namespace rtl {

// A field in a struct or union
struct FieldInfo {
  llvm::StringRef name;
  mlir::Type type;

public:
  FieldInfo(llvm::StringRef name, mlir::Type type) : name(name), type(type) {}

  FieldInfo allocateInto(::mlir::TypeStorageAllocator &alloc) const {
    llvm::StringRef nameCopy = alloc.copyInto(name);
    return FieldInfo(nameCopy, type);
  }
};

} // namespace rtl
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTL/RTLTypes.h.inc"

#endif
