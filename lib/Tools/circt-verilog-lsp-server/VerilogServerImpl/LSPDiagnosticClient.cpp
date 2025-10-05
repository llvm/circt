//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A converter that can be plugged into a slang `DiagnosticEngine` as a
// client that will map slang diagnostics to LSP diagnostics.
//
//===----------------------------------------------------------------------===//

#include "LSPDiagnosticClient.h"

using namespace circt::lsp;

static llvm::lsp::DiagnosticSeverity
getSeverity(slang::DiagnosticSeverity severity) {
  switch (severity) {
  case slang::DiagnosticSeverity::Fatal:
  case slang::DiagnosticSeverity::Error:
    return llvm::lsp::DiagnosticSeverity::Error;
  case slang::DiagnosticSeverity::Warning:
    return llvm::lsp::DiagnosticSeverity::Warning;
  case slang::DiagnosticSeverity::Ignored:
  case slang::DiagnosticSeverity::Note:
    return llvm::lsp::DiagnosticSeverity::Information;
  }
  llvm_unreachable("all slang diagnostic severities should be handled");
  return llvm::lsp::DiagnosticSeverity::Error;
}

void LSPDiagnosticClient::report(const slang::ReportedDiagnostic &slangDiag) {
  auto loc = document.getLspLocation(slangDiag.location);
  // Show only the diagnostics in the current file.
  if (loc.uri != document.getURI())
    return;
  auto &mlirDiag = diags.emplace_back();
  mlirDiag.severity = getSeverity(slangDiag.severity);
  mlirDiag.range = loc.range;
  mlirDiag.source = "slang";
  mlirDiag.message = slangDiag.formattedMessage;
}
