//===- EmitOpInterfaces.cpp - Implement the Emit op interfaces ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the Emit operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitOpInterfaces.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace llvm;

#include "circt/Dialect/Emit/EmitOpInterfaces.cpp.inc"
