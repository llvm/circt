// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __ESI_TYPES_HPP__
#define __ESI_TYPES_HPP__

#include <tuple>

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>

namespace mlir {
namespace esi {

struct FieldInfo {
  llvm::StringRef name;
  Type type;

public:
  FieldInfo(StringRef name, Type type) : name(name), type(type) {}

  FieldInfo allocateInto(::mlir::TypeStorageAllocator &alloc) const {
    StringRef nameCopy = alloc.copyInto(name);
    return FieldInfo(nameCopy, type);
  }
};

} // namespace esi
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.h.inc"

#endif
