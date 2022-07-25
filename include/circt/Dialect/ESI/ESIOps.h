//===- ESIOps.h - ESI operations --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI Ops are defined in tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIOPS_H
#define CIRCT_DIALECT_ESI_ESIOPS_H

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESITypes.h"

#include "circt/Dialect/HW/HWAttributes.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/ESI/ESIInterfaces.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.h.inc"

#endif
