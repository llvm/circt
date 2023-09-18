//===- InstanceGraphInterface.h - Instance graph interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares stuff related to the instance graph interface.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_INSTANCEGRAPHINTERFACE_H
#define CIRCT_SUPPORT_INSTANCEGRAPHINTERFACE_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

/// The InstanceGraph op interface, see InstanceGraphInterface.td for more
/// details.
#include "circt/Support/InstanceGraphInterface.h.inc"

#endif // CIRCT_SUPPORT_INSTANCEGRAPHINTERFACE_H
