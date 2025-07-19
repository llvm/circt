//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Synthesis.h"

#include "circt/Synthesis/Transforms/Passes.h"
#include "circt/Synthesis/Transforms/SynthesisPipeline.h"

void registerSynthesisPipeline() {
  circt::synthesis::registerPipelines();
  circt::synthesis::registerPasses();
}
