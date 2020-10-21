//===- ESITypes.h - types for the ESI dialect -------------------*- C++ -*-===//
//
// Types for ESI are mostly in tablegen. This file should contain C++ types used
// in MLIR type parameters.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESITYPES_H
#define CIRCT_DIALECT_ESI_ESITYPES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "ESIDialect.h"

namespace circt {
namespace esi {

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

} // namespace esi
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.h.inc"

#endif
