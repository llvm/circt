//===- FIRRTLAttributes.h - FIRRTL dialect attributes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the FIRRTL dialect custom attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_OFIRATTRIBUTES_H
#define CIRCT_DIALECT_FIRRTL_OFIRATTRIBUTES_H

#include "circt/Dialect/OFIR/OFIRDialect.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace ofir {

} // namespace ofir
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/OFIR/OFIRAttributes.h.inc"

#endif // CIRCT_DIALECT_OFIR_OFIRATTRIBUTES_H
