//===- ArcilatorHeaders.cpp - Arcilator headers generator -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates header files to link to the output of the `arcilator` compiler.
//
//===----------------------------------------------------------------------===//

#include "ArcilatorHeaders.h"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace arcilator;

mlir::LogicalResult generateHeaders(llvm::raw_ostream &output,
                                    const std::vector<StateInfo> &states) {
                                    
  return failure();
}
