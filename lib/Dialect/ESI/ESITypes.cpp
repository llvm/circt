// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "circt/Dialect/ESI/ESITypes.hpp"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/TableGen/TypeDefGenHelpers.h>

using namespace mlir;

namespace mlir {
namespace esi {

static bool operator==(const FieldInfo &a, const FieldInfo &b) {
  return a.name == b.name && a.type == b.type;
}

static llvm::hash_code hash_value(const FieldInfo &fi) {
  return llvm::hash_combine(fi.name, fi.type);
}

} // namespace esi
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.cpp.inc"
