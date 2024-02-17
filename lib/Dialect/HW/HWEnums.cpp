//===- HWEnums.cpp - Implement the HW enums -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HW enums.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWEnums.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace circt;
using namespace hw;

// Provide implementations for the enums we use.
#include "circt/Dialect/HW/HWEnums.cpp.inc"
