// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __ESI_TYPES_HPP__
#define __ESI_TYPES_HPP__

#include <tuple>

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>

namespace circt {
namespace esi {

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
