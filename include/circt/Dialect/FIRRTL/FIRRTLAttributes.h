//===- FIRRTLAttributes.h - FIRRTL dialect attributes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the FIRRTL dialect custom attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_H

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLEnums.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Support/LLVM.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h.inc"

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_H
