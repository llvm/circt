//===- SSPAttributes.h - SSP attribute definitions --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SSP (static scheduling problem) dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SSP_SSPATTRIBUTES_H
#define CIRCT_DIALECT_SSP_SSPATTRIBUTES_H

#include "circt/Dialect/SSP/SSPDialect.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SSP/SSPAttributes.h.inc"

#endif // CIRCT_DIALECT_SSP_SSPATTRIBUTES_H
