//===- circt-verilog-lsp.cpp - Getting Verilog into CIRCT -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a utility to parse Verilog and SystemVerilog input
// files. This builds on CIRCT's ImportVerilog library, which ultimately relies
// on slang to do the heavy lifting.
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-verilog-lsp/CirctVerilogLspServerMain.h"

#include "llvm/Support/LogicalResult.h"

int main(int argc, char **argv) {
  return failed(circt::CirctVerilogLspServerMain(argc, argv));
}
