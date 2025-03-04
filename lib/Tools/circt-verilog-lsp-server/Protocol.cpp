//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the serialization code for the Verilog specific LSP
// structs.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "llvm/Support/JSON.h"

using namespace circt::lsp;

//===----------------------------------------------------------------------===//
// VerilogUserProvidedInlayHintParams
//===----------------------------------------------------------------------===//

bool circt::lsp::fromJSON(const llvm::json::Value &value,
                          VerilogUserProvidedInlayHint &result,
                          llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  if (!o)
    return false;

  if (!o.map("path", result.path) || !o.map("value", result.value))
    return false;

  (void)o.map("root", result.root);
  (void)o.map("group", result.group);
  return true;
}

bool circt::lsp::fromJSON(const llvm::json::Value &value,
                          VerilogUserProvidedInlayHintParams &result,
                          llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("hints", result.hints);
}
