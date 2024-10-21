//===- EmitRTGAssembly.h - RTG Assembly Exporter ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the Assembly emitter for the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_EMITRTGASSEMBLY_H
#define CIRCT_TARGET_EMITRTGASSEMBLY_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace EmitRTGAssembly {

/// Emission options for the EmitRTGAssembly pass. Allows controlling the
/// emitted format and overall behavior.
struct EmitRTGAssemblyOptions {
  // List of operations for which the textual format is used.
  SmallVector<std::string> unsupportedInstructions;
};

/// Run the EmitRTGAssembly pass.
LogicalResult emitRTGAssembly(
    Operation *module, llvm::raw_ostream &os,
    const EmitRTGAssemblyOptions &options = EmitRTGAssemblyOptions());

/// Parse the given file for unsupported instructions.
void parseUnsupportedInstructionsFile(
    const std::string &unsupportedInstructionsFile,
    SmallVectorImpl<std::string> &unsupportedInstrs);

/// Register the EmitRTGAssembly pass.
void registerEmitRTGAssemblyTranslation();

} // namespace EmitRTGAssembly
} // namespace circt

#endif // CIRCT_TARGET_EMITRTGASSEMBLY_H
