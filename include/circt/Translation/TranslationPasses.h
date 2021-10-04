//===- TranslationPasses.h - Translation pass entry points ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TRANSLATION_TRANSLATIONPASSES_H
#define CIRCT_TRANSLATION_TRANSLATIONPASSES_H

#include "mlir/Pass/Pass.h"

namespace circt {
namespace translations {

std::unique_ptr<mlir::Pass> createExportVerilogFilePass(llvm::raw_ostream &os);
std::unique_ptr<mlir::Pass> createExportVerilogFilePass() {
  return createExportVerilogFilePass(llvm::outs());
}

std::unique_ptr<mlir::Pass>
createExportSplitVerilogPass(StringRef directory = "./");

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Translation/TranslationPasses.h.inc"

} // namespace translations
} // namespace circt

#endif // CIRCT_TRANSLATION_TRANSLATIONPASSES_H
