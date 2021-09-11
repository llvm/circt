//===- ComponentLibrary.h - Library of components ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a library of components that can be scheduled.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_COMPONENTLIBRARY_H
#define CIRCT_SCHEDULING_COMPONENTLIBRARY_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

namespace circt {
namespace scheduling {

struct ComponentData {
  ComponentData(unsigned latency) : latency(latency) {}
  unsigned latency;
};

struct ComponentLibrary {
  template <typename SourceOp>
  void addComponent(unsigned latency) {
    components.insert(
        std::pair(SourceOp::getOperationName(), ComponentData(latency)));
  }
  llvm::Optional<unsigned> getLatency(Operation *);

private:
  llvm::StringMap<ComponentData> components;
};

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_COMPONENTLIBRARY_H
