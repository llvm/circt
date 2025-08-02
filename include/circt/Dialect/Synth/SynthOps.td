//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations of the Synth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHOPS_TD
#define CIRCT_DIALECT_SYNTH_SYNTHOPS_TD

include "circt/Dialect/Synth/Synth.td"
include "mlir/IR/OpBase.td"

// Base class for the operations in this dialect.
class SynthOp<string mnemonic, list<Trait> traits = []> :
    Op<Synth_Dialect, mnemonic, traits>;

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPS_TD
