//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_LSPDIAGNOSTICCLIENT_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_LSPDIAGNOSTICCLIENT_H_

#include "VerilogDocument.h"

namespace circt {
namespace lsp {

class LSPDiagnosticClient : public slang::DiagnosticClient {
  const VerilogDocument &document;
  std::vector<llvm::lsp::Diagnostic> &diags;

public:
  LSPDiagnosticClient(const VerilogDocument &document,
                      std::vector<llvm::lsp::Diagnostic> &diags)
      : document(document), diags(diags) {}

  void report(const slang::ReportedDiagnostic &slangDiag) override;
};
} // namespace lsp
} // namespace circt

#endif
