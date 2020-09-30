//===- ESITypes.h - types for the ESI dialect -------------------*- C++ -*-===//
//
// Types for ESI are mostly in tablegen. This file should contain C++ types used
// in MLIR type parameters.
//
//===----------------------------------------------------------------------===//

#ifndef __ESI_TYPES_HPP__
#define __ESI_TYPES_HPP__

#include <tuple>

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>

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

// Symbols for the string type encoding
enum StringEncoding { ASCII, UTF8, UTF16, UTF32 };

} // namespace esi
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.h.inc"

#endif
