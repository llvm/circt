//===- SynthDialect.td - Synth dialect definition ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the Synth dialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHDIALECT
#define CIRCT_DIALECT_SYNTH_SYNTHDIALECT

include "mlir/IR/DialectBase.td"

def Synth_Dialect : Dialect {
  let name = "synth";
  let cppNamespace = "::circt::synth";
  let summary = "Synthesis dialect for logic synthesis operations";
  let description = [{
    The Synth dialect provides operations and types for logic synthesis,
    including meta operations for synthesis decisions, logic representations
    like AIG, and synthesis pipeline infrastructure.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // CIRCT_DIALECT_SYNTH_SYNTHDIALECT
