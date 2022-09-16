//===- PassDetails.h - ESI pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different ESI passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_ESI_PASSDETAILS_H
#define DIALECT_ESI_PASSDETAILS_H

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace esi {

#define GEN_PASS_CLASSES
#include "circt/Dialect/ESI/ESIPasses.h.inc"

} // namespace esi
} // namespace circt

#endif // DIALECT_ESI_PASSDETAILS_H
