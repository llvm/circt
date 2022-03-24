//===- CHALKTypes.cpp - Implementation of CHALK dialect types -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CHALK/CHALKTypes.h"
#include "circt/Dialect/CHALK/CHALKOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace chalk;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/CHALK/CHALKTypes.cpp.inc"
