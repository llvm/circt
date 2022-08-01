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

#ifndef CIRCT_TARGET_EXPORTSYSTEMC_H
#define CIRCT_TARGET_EXPORTSYSTEMC_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace ExportSystemC {

class EmissionConfig;

LogicalResult exportSystemC(ModuleOp module, EmissionConfig &config,
                            llvm::raw_ostream &os);
LogicalResult exportSystemC(ModuleOp module, EmissionConfig &config);

LogicalResult exportSplitSystemC(ModuleOp module, EmissionConfig &config,
                                 llvm::StringRef directory = "./");

void registerExportSystemCTranslation();

} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_H
