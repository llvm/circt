//===- ComponentLibrary.cpp - Library of components -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a library of components that can be scheduled.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/ComponentLibrary.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace circt::scheduling;

Optional<unsigned>
circt::scheduling::ComponentLibrary::getLatency(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  auto it = components.find(opName);
  if (it != components.end())
    return it->second.latency;

  return llvm::None;
}
