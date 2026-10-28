//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_LLHDTYPES_H
#define CIRCT_DIALECT_LLHD_LLHDTYPES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

namespace circt {
namespace llhd {

class NetType;
class RefType;

/// Common base of types wrapping a nested type.
class WrapperType : public mlir::Type {
public:
  static bool classof(mlir::Type type) {
    return llvm::isa<RefType, NetType>(type);
  }

  /// Get the wrapped type.
  mlir::Type getNestedType() const;

protected:
  using mlir::Type::Type;
};

} // namespace llhd
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/LLHD/LLHDTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/LLHD/LLHDAttributes.h.inc"

namespace llvm {
template <>
struct PointerLikeTypeTraits<circt::llhd::TimeAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline circt::llhd::TimeAttr getFromVoidPointer(void *p) {
    return circt::llhd::TimeAttr::getFromOpaquePointer(p);
  }
};
} // namespace llvm

#endif // CIRCT_DIALECT_LLHD_LLHDTYPES_H
