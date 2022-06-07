//===- GAAOpInterfaces.cpp - Implement the GAA op interfaces --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the GAA operation interfaces.
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/GAA/GAAAttributes.h"
#include "circt/Dialect/GAA/GAAOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace llvm;
using namespace circt::gaa;
#include "circt/Dialect/GAA/GAAOpInterfaces.cpp.inc"