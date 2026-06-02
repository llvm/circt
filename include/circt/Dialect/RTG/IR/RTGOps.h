//===- RTGOps.h - Declare RTG dialect operations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGOPS_H
#define CIRCT_DIALECT_RTG_IR_RTGOPS_H

#include "circt/Dialect/Emit/EmitOpInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGAttrInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyAttrInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyOpInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyTypeInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGTypeInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace rtg {

/// A custom resource representing the RNG state. Operations that read from or
/// write to the RNG state should declare effects on this resource to prevent
/// CSE and DCE.
struct RNGStateResource
    : public mlir::SideEffects::Resource::Base<RNGStateResource> {
  StringRef getName() const final { return "RTGRNGStateResource"; }
  bool isAddressable() const override { return false; }
};

} // namespace rtg
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/RTG/IR/RTG.h.inc"

#endif // CIRCT_DIALECT_RTG_IR_RTGOPS_H
