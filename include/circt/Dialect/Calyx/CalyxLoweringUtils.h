//===- CalyxLoweringUtils.h - Calyx lowering utility methods ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines various lowering utility methods for converting to
// and from Calyx programs.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H
#define CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Support/LLVM.h"

#include <variant>

namespace circt {
namespace calyx {

// A structure representing a set of ports which act as a memory interface for
// external memories.
struct MemoryPortsImpl {
  Value readData;
  Value done;
  Value writeData;
  SmallVector<Value> addrPorts;
  Value writeEn;
};

// Represents the interface of memory in Calyx. The various lowering passes
// are agnostic wrt. whether working with a calyx::MemoryOp (internally
// allocated memory) or MemoryPortsImpl (external memory).
struct MemoryInterface {
  MemoryInterface();
  explicit MemoryInterface(const MemoryPortsImpl &ports);
  explicit MemoryInterface(calyx::MemoryOp memOp);

  // Getter methods for each memory interface port.
  Value readData();
  Value done();
  Value writeData();
  Value writeEn();
  ValueRange addrPorts();

private:
  std::variant<calyx::MemoryOp, MemoryPortsImpl> impl;
};

} // namespace calyx
} // namespace circt

#endif // CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H
