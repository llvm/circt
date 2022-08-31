//===- SystemCTypes.h - Declare SystemC dialect types ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES_H
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES_H

#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

namespace circt {
namespace systemc {

namespace detail {
/// A struct containing minimal information for a systemc module port. Thus, can
/// be used as parameter in attributes or types.
struct PortInfo {
  mlir::StringAttr name;
  mlir::Type type;
};
} // namespace detail

/// Get the type wrapped by a signal or port (in, inout, out) type.
Type getSignalBaseType(Type type);

} // namespace systemc
} // namespace circt

// Include generated types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SystemC/SystemCTypes.h.inc"

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES_H
