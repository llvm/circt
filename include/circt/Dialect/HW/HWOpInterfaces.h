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

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace circt {
namespace hw {
struct PortInfo;
} // namespace hw
} // namespace circt

#include "circt/Dialect/HW/HWOpInterfaces.h.inc"

#endif // CIRCT_DIALECT_HW_HWOPINTERFACES_H
