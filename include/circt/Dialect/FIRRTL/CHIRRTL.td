//===- CHIRRTL.td - CHIRRTL dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the CHIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CHIRRTL_TD
#define CHIRRTL_TD

// Dialect
include "circt/Dialect/FIRRTL/CHIRRTLDialect.td"

// Operations
include "circt/Dialect/FIRRTL/CHIRRTLOps.td"

// Types
include "circt/Dialect/FIRRTL/CHIRRTLTypes.td"

#endif // CHIRRTL_TD
