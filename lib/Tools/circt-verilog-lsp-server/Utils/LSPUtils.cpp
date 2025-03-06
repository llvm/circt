//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities for CIRCT Verilog LSP server.
//
//===----------------------------------------------------------------------===//

#include "LSPUtils.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
void circt::lsp::Logger::error(Twine message) {
  mlir::lsp::Logger::error("{}", message);
}

void circt::lsp::Logger::info(Twine message) {
  mlir::lsp::Logger::info("{}", message);
}

void circt::lsp::Logger::debug(Twine message) {
  mlir::lsp::Logger::debug("{}", message);
}

void circt::lsp::printReindented(llvm::raw_ostream &os, StringRef content) {
  mlir::raw_indented_ostream indentedOs(os);
  indentedOs.printReindented(content);
}
