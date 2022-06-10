//===- FIRRTLOpInterfaces.cpp - Implement the HW op interfaces ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the HW operation interfaces.
//
//===----------------------------------------------------------------------===//

#include <circt/Dialect/HW/HWOpInterfaces.h>
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"

#include "circt/Dialect/HW/HWOpInterfaces.cpp.inc"
circt::hw::HWInstanceLike getInstance(mlir::SymbolTable &symtbl,
                                                 circt::hw::InnerRefAttr name){
    auto mod = symtbl.lookup<circt::hw::HWModuleLike>(name.getModule());
    if (!mod)
      return {};
    return mod.getInstance(name.getName());
}

