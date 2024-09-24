//===- EmitAssembly.h - Assembly Exporter -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the Assembly emitter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_EMITASSEMBLY_H
#define CIRCT_TARGET_EMITASSEMBLY_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace EmitAssembly {

/// Emission options for the EmitAssembly pass. Allows controlling the emitted
/// format and overall behavior.
struct EmitAssemblyOptions {
  // List of operations for which the textual format is used.
  SmallVector<std::string> supportedInstructions;
};

/// Run the EmitAssembly pass.
LogicalResult
emitAssembly(Operation *module, llvm::raw_ostream &os,
             const EmitAssemblyOptions &options = EmitAssemblyOptions());

/// Register the EmitAssembly pass.
void registerEmitAssemblyTranslation();

} // namespace EmitAssembly
} // namespace circt

#endif // CIRCT_TARGET_EMITASSEMBLY_H
