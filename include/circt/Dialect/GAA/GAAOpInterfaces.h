//===- GAAOpInterfaces.h - Declare GAA op interfaces ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the GAA IR and supporting
// types.
//
//===----------------------------------------------------------------------===//
#ifndef CIRCT_DIALECT_GAA_GAAOPINTERFAES_H
#define CIRCT_DIALECT_GAA_GAAOPINTERFAES_H

#include "llvm/Support/CommandLine.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/GAA/GAAOpInterfaces.h.inc"

#endif // CIRCT_DIALECT_GAA_GAAOPINTERFAES_H