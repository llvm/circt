//===- FIRRTLOps.h - Declare FIRRTL dialect operations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation class for the OFIR IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OFIR_OPS_H
#define CIRCT_DIALECT_OFIR_OPS_H

#include "circt/Dialect/OFIR/OFIRDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"


namespace circt {
namespace ofir {

} // namespace ofir
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/OFIR/OFIR.h.inc"

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

#endif // CIRCT_DIALECT_OFIR_OPS_H
