//===- HWUtils.h - HW Rewriting Utilities -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helper utilities to transform HW IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_HWUTILS_H
#define CIRCT_SUPPORT_HWUTILS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"

namespace circt {

/// Append input and output ports to a module.
/// Use `ports` as the PortInfo for the new ports to add.
/// Use `outputVals` as the values for the new output ports.
/// Add all the new input block arguments to `inputVals`.
void appendPorts(hw::HWModuleOp module, ArrayRef<hw::PortInfo> ports,
                 ArrayRef<Value> outputVals, SmallVectorImpl<Value> &inputVals);

/// Remove ports from a module based on a predicate function.
/// The predicate function takes a PortInfo and returns true if the port
/// should be removed. Updates all corresponding attributes after dropping the
/// ports and updates all instances of the module.
/// `shouldRemove` - predicate to determine if a port should be removed
/// `dropModuleArg` - callback invoked for each removed input block argument
/// `dropResult` - callback invoked for each removed output result
void removePorts(hw::HWModuleOp module, igraph::InstanceGraph &instanceGraph,
                 const std::function<bool(const hw::PortInfo &)> &shouldRemove,
                 const std::function<bool(BlockArgument)> &dropModuleArg,
                 const std::function<bool(Operation *, unsigned)> &dropResult);

/// Remove ports from an extern module based on a predicate function.
/// The predicate function takes a PortInfo and returns true if the port
/// should be removed. Updates all corresponding attributes after dropping the
/// ports and updates all instances of the module.
/// `shouldRemove` - predicate to determine if a port should be removed
/// `dropResult` - callback invoked for each removed output result
void removePorts(hw::HWModuleExternOp module,
                 igraph::InstanceGraph &instanceGraph,
                 const std::function<bool(const hw::PortInfo &)> &shouldRemove,
                 const std::function<bool(Operation *, unsigned)> &dropResult);

} // namespace circt

#endif // CIRCT_SUPPORT_HWUTILS_H
