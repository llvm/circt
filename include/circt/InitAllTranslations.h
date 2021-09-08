//===- InitAllTranslations.h - CIRCT Translations Registration --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all translations
// in and out of CIRCT to the system.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxEmitter.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/FIRRTL/FIREmitter.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/LLHD/Translation/TranslateToVerilog.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Translation/ExportVerilog.h"
//#include "circt/Translation/HIRToVerilog.h"

#ifndef CIRCT_INITALLTRANSLATIONS_H
#define CIRCT_INITALLTRANSLATIONS_H

namespace circt {

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerAllTranslations() {
  static bool initOnce = []() {
    // mlir::hir::registerHIRToVerilogTranslation();
    registerToVerilogTranslation();
    esi::registerESITranslations();
    calyx::registerToCalyxTranslation();
    firrtl::registerFromFIRFileTranslation();
    firrtl::registerToFIRFileTranslation();
    llhd::registerToVerilogTranslation();
    msft::registerMSFTTclTranslation();
    return true;
  }();
  (void)initOnce;
}
} // namespace circt

#endif // CIRCT_INITALLTRANSLATIONS_H
