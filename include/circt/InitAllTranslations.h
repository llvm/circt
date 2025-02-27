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

#include "circt/Dialect/Arc/ModelInfoExport.h"
#include "circt/Dialect/BLIF/BLIFEmitter.h"
#include "circt/Dialect/BLIF/BLIFParser.h"
#include "circt/Dialect/Calyx/CalyxEmitter.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/FIRRTL/FIREmitter.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Target/DebugInfo.h"
#include "circt/Target/ExportSMTLIB.h"
#include "circt/Target/ExportSystemC.h"

#ifndef CIRCT_INITALLTRANSLATIONS_H
#define CIRCT_INITALLTRANSLATIONS_H

namespace circt {

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerAllTranslations() {
  static bool initOnce = []() {
    arc::registerArcModelInfoTranslation();
    blif::registerFromBLIFFileTranslation();
    blif::registerToBLIFFileTranslation();
    calyx::registerToCalyxTranslation();
    firrtl::registerFromFIRFileTranslation();
    firrtl::registerToFIRFileTranslation();
    ExportSMTLIB::registerExportSMTLIBTranslation();
    ExportSystemC::registerExportSystemCTranslation();
    debug::registerTranslations();
    return true;
  }();
  (void)initOnce;
}
} // namespace circt

#endif // CIRCT_INITALLTRANSLATIONS_H
