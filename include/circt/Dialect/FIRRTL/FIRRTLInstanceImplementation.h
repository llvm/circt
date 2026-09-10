//===- FIRRTLInstanceImplementation.h - Instance-like utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utility functions for implementing FIRRTL instance-like
// operations, in particular, parsing, and printing common to instance-like
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLINSTANCEIMPLEMENTATION_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLINSTANCEIMPLEMENTATION_H

#include "circt/Support/LLVM.h"

namespace circt {
namespace firrtl {
namespace instance_like_impl {

/// Verify that the instance refers to a valid FIRRTL module.
LogicalResult verifyReferencedModule(Operation *instanceOp,
                                     SymbolTableCollection &symbolTable,
                                     mlir::FlatSymbolRefAttr moduleName);

} // namespace instance_like_impl
} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLINSTANCEIMPLEMENTATION_H
