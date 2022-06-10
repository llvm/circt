//===- HWOpInterfaces.h - Declare HW op interfaces --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWOPINTERFACES_H
#define CIRCT_DIALECT_HW_HWOPINTERFACES_H

#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace hw {
class HWInstanceLike;
class HWModuleLike;
}
}

#include "circt/Dialect/HW/HWOpInterfaces.h.inc"

circt::hw::HWInstanceLike getInstance(mlir::SymbolTable &symtbl, circt::hw::InnerRefAttr name);

#endif // CIRCT_DIALECT_HW_HWOPINTERFACES_H
