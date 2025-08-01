//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Synth.h"

#include "circt/Dialect/Synth/Transforms/SynthesisPipeline.h"

void registerSynthesisPipeline() { circt::synth::registerSynthesisPipeline(); }
