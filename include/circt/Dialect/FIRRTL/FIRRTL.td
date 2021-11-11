//===- FIRRTL.td - FIRRTL dialect definition ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FIRRTL_TD
#define FIRRTL_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"

include "FIRRTLDialect.td"
include "FIRRTLTypes.td"
include "FIRRTLOpInterfaces.td"
include "FIRRTLStructure.td"
include "FIRRTLDeclarations.td"
include "FIRRTLExpressions.td"
include "FIRRTLStatements.td"

#endif // FIRRTL_TD
