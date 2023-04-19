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

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTL_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTL_TD

// Dialect
include "circt/Dialect/FIRRTL/FIRRTLDialect.td"

// Operations
include "circt/Dialect/FIRRTL/FIRRTLStructure.td"
include "circt/Dialect/FIRRTL/FIRRTLDeclarations.td"
include "circt/Dialect/FIRRTL/FIRRTLStatements.td"
include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

// Types
include "circt/Dialect/FIRRTL/FIRRTLTypes.td"
include "circt/Dialect/FIRRTL/FIRRTLTypesImpl.td"

// Attributes
include "circt/Dialect/FIRRTL/FIRRTLAttributes.td"

// Op Interfaces
include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.td"

#endif // CIRCT_DIALECT_FIRRTL_FIRRTL_TD
