//===- ExportSystemC.h - SystemC Exporter -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the SystemC emitter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TRANSLATION_EXPORTSYSTEMC_H
#define CIRCT_TRANSLATION_EXPORTSYSTEMC_H

#include "mlir/Pass/Pass.h"

namespace circt {

std::unique_ptr<mlir::Pass> createExportSystemCPass(llvm::raw_ostream &os);
std::unique_ptr<mlir::Pass> createExportSystemCPass();

std::unique_ptr<mlir::Pass>
createExportSplitSystemCPass(llvm::StringRef directory = "./");

} // namespace circt

#endif // CIRCT_TRANSLATION_EXPORTSYSTEMC_H
