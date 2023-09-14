//===- AppID.cpp - AppID related code -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/AppID.h"

using namespace circt;
using namespace msft;

AppIDIndex::AppIDIndex(Operation *root) : root(root) {}

DynamicInstanceOp AppIDIndex::getInstance(AppIDPathAttr path) {
  return nullptr;
}
