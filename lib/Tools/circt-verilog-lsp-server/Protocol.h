//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains structs for LSP commands that are specific to the Verilog
// server.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCTVERILOGSPLSERVER_PROTOCOL_H_
#define LIB_CIRCT_TOOLS_CIRCTVERILOGSPLSERVER_PROTOCOL_H_

#include "llvm/Support/JSON.h"
#include <optional>
#include <string>
#include <vector>

namespace circt {
namespace lsp {

//===----------------------------------------------------------------------===//
// VerilogUserProvidedInlayHint
//===----------------------------------------------------------------------===//

struct VerilogUserProvidedInlayHint {
  // The object path to the value.
  std::string path;

  // The value.
  std::string value;

  // The root module name (optional).
  std::optional<std::string> root;

  // The optional id of the hint.
  std::optional<std::string> group;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              VerilogUserProvidedInlayHint &result, llvm::json::Path path);

struct VerilogUserProvidedInlayHintParams {
  std::vector<VerilogUserProvidedInlayHint> hints;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              VerilogUserProvidedInlayHintParams &result,
              llvm::json::Path path);

} // namespace lsp
} // namespace circt

#endif
