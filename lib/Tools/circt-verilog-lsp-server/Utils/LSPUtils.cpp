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
#include "llvm/Support/LSP/Logging.h"

void circt::lsp::Logger::error(Twine message) {
  llvm::lsp::Logger::error("{}", message);
}

void circt::lsp::Logger::info(Twine message) {
  llvm::lsp::Logger::info("{}", message);
}

void circt::lsp::Logger::debug(Twine message) {
  llvm::lsp::Logger::debug("{}", message);
}
