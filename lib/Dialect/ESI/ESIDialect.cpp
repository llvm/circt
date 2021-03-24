//===- ESIDialect.cpp - ESI dialect code defs -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialect definitions. Should be relatively standard boilerplate.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/Support/FormatVariadic.h"

namespace circt {
namespace esi {

ESIDialect::ESIDialect(MLIRContext *context)
    : Dialect("esi", context, TypeID::get<ESIDialect>()) {

  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/ESI/ESI.cpp.inc"
      >();
}
} // namespace esi
} // namespace circt

#include "circt/Dialect/ESI/ESIAttrs.cpp.inc"
