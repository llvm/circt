//===- FIRRTLToHW.h - FIRRTL to HW conversion pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the FIRRTL dialect to
// HW and SV dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FIRRTLTOHW_FIRRTLTOHW_H
#define CIRCT_CONVERSION_FIRRTLTOHW_FIRRTLTOHW_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace firrtl {
enum class VerificationFlavor {
  // Use the flavor specified by the op.
  // TOOD: Drop this option once the migration finished.
  None,
  // Use `if(cond) else $fatal(..)` format.
  IfElseFatal,
  // Use immediate form.
  Immediate,
  // Use SVA.
  SVA
};
} // namespace firrtl

#define GEN_PASS_DECL_LOWERFIRRTLTOHW
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass>
createLowerFIRRTLToHWPass(bool enableAnnotationWarning = false,
                          firrtl::VerificationFlavor assertionFlavor =
                              firrtl::VerificationFlavor::None);

} // namespace circt

#endif // CIRCT_CONVERSION_FIRRTLTOHW_FIRRTLTOHW_H
