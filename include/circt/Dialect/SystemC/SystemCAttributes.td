//===- SystemCAttributes.td - SystemC attributes -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Attribute definitions for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCATTRIBUTES_TD
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCATTRIBUTES_TD

include "circt/Dialect/SystemC/SystemCDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class SystemCAttrDef<string name> : AttrDef<SystemCDialect, name> { }

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCATTRIBUTES_TD
