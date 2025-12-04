//===- RTGAttributes.h - RTG dialect attributes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGATTRIBUTES_H
#define CIRCT_DIALECT_RTG_IR_RTGATTRIBUTES_H

#include "circt/Dialect/RTG/IR/RTGAttrInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyAttrInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace circt {
namespace rtg {
namespace detail {

struct ImmediateAttrStorage;

} // namespace detail
} // namespace rtg
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/RTG/IR/RTGAttributes.h.inc"

#endif // CIRCT_DIALECT_RTG_IR_RTGATTRIBUTES_H
