//===- PassDetails.h - Seq pass class details -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different Seq passes.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_SEQ_TRANSFORMS_PASSDETAILS_H
#define DIALECT_SEQ_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace seq {

#define GEN_PASS_DEF_LOWERSEQFIRRTLTOSV
#define GEN_PASS_DEF_LOWERSEQHLMEM
#define GEN_PASS_DEF_LOWERSEQTOSV
#include "circt/Dialect/Seq/SeqPasses.h.inc"

} // namespace seq
} // namespace circt

#endif // DIALECT_SEQ_TRANSFORMS_PASSDETAILS_H
