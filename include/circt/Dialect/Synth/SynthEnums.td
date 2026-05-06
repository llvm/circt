//===- SynthEnums.td - Enums for Synth dialect -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Synth dialect specific enums.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHENUMS_TD
#define CIRCT_DIALECT_SYNTH_SYNTHENUMS_TD

include "mlir/IR/EnumAttr.td"

let cppNamespace = "circt::synth" in {

def Polarity_Positive : I32EnumAttrCase<"Positive", 0, "positive">;
def Polarity_Negative : I32EnumAttrCase<"Negative", 1, "negative">;

def PolarityKindAttr : I32EnumAttr<"PolarityKind", "timing arc polarity",
  [Polarity_Positive, Polarity_Negative]>;

}

#endif // CIRCT_DIALECT_SYNTH_SYNTHENUMS_TD
