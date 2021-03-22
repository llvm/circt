//===- MSFTAttributes.h - Microsoft dialect attributes ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSFT dialect custom attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_MSFTATTRIBUTES_H
#define CIRCT_DIALECT_MSFT_MSFTATTRIBUTES_H

#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace msft {

namespace detail {
struct PhysLocationAttrStorage;
}

class PhysLocationAttr
    : public Attribute::AttrBase<PhysLocationAttr, Attribute,
                                 detail::PhysLocationAttrStorage> {
public:
  using Base::Base;
  static PhysLocationAttr get(DeviceTypeAttr Type, uint64_t X, uint64_t Y,
                              uint64_t Num, StringRef Entity);

  DeviceTypeAttr Type() const;
  uint64_t X() const;
  uint64_t Y() const;
  uint64_t Num() const;
  StringRef Entity() const;

  /// Parse a PhysLocationAttr with the following syntax:
  /// #msft.physloc<DevType, X, Y, Num, "Entity">
  static Attribute parse(DialectAsmParser &p);
  void print(DialectAsmPrinter &) const;
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_MSFTATTRIBUTES_H
