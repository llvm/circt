//===--- Protocol.cpp - Language Server Protocol Implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the serialization code for the PDLL specific LSP structs.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::lsp;

// Helper that doesn't treat `null` and absent fields as failures.
template <typename T>
static bool mapOptOrNull(const llvm::json::Value &params,
                         llvm::StringLiteral prop, T &out,
                         llvm::json::Path path) {
  const llvm::json::Object *o = params.getAsObject();
  assert(o);

  // Field is missing or null.
  auto *v = o->get(prop);
  if (!v || v->getAsNull())
    return true;
  return fromJSON(*v, out, path.field(prop));
}

//===----------------------------------------------------------------------===//
// VerilogViewOutputParams
//===----------------------------------------------------------------------===//

bool circt::lsp::fromJSON(const llvm::json::Value &value,
                         VerilogViewOutputKind &result, llvm::json::Path path) {
  if (std::optional<StringRef> str = value.getAsString()) {
    if (*str == "ast") {
      result = VerilogViewOutputKind::AST;
      return true;
    }
    if (*str == "mlir") {
      result = VerilogViewOutputKind::MLIR;
      return true;
    }
    if (*str == "cpp") {
      result = VerilogViewOutputKind::CPP;
      return true;
    }
  }
  return false;
}

bool circt::lsp::fromJSON(const llvm::json::Value &value,
                         VerilogViewOutputParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("uri", result.uri) && o.map("kind", result.kind);
}

//===----------------------------------------------------------------------===//
// VerilogViewOutputResult
//===----------------------------------------------------------------------===//

llvm::json::Value circt::lsp::toJSON(const VerilogViewOutputResult &value) {
  return llvm::json::Object{{"output", value.output}};
}
