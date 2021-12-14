//===- Types.td - Common type predicates and definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file collects type predicates and definitions commonly used in the CIRCT
// dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TYPES
#define CIRCT_TYPES

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def APSIntAttr : Attr<CPred<"$_self.isa<::mlir::IntegerAttr>()">,
                      "arbitrary integer attribute with sign"> {
  let storageType = [{ ::mlir::IntegerAttr }];
  let returnType = [{ ::llvm::APSInt }];
  let constBuilderCall = "IntegerAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "APSInt($_self.getValue(), !getType().isSigned())";
}

#endif // CIRCT_TYPES
