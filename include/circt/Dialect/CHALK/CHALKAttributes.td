//===- CHALKAttributes.td - CHALK Types Definition -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of CHALK attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_CHALKATTRIBUTES_TD
#define CIRCT_DIALECT_CHALK_CHALKATTRIBUTES_TD

//===----------------------------------------------------------------------===//
// CHALK Annotations Definition
//===----------------------------------------------------------------------===//

def CellIntAttr: ArrayAttrBase< And<[CPred<"$_self.isa<::mlir::ArrayAttr>()">]>, ""> { }
// def CellIntAttr: AttrConstraint<CPred<"$_self.isa<IntegerAttr>()">, "Integer Type"> { }
// def HasSomeProperty : AnyIntegerAttrBase< And<[CPred<"$_self.isa<::mlir::IntegerAttr>()">]>, "Integer type">;

#endif // CIRCT_DIALECT_CHALK_CHALKATTRIBUTES_TD
