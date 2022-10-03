//===- LoweringOptionsParser.h - CIRCT Lowering Option Parser ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Parser for lowering options.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_LOWERINGOPTIONSPARSER_H
#define CIRCT_SUPPORT_LOWERINGOPTIONSPARSER_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace circt {

/// Commandline parser for LoweringOptions.  Delegates to the parser
/// defined by LoweringOptions.
struct LoweringOptionsParser : public llvm::cl::parser<LoweringOptions> {

  LoweringOptionsParser(llvm::cl::Option &option)
      : llvm::cl::parser<LoweringOptions>(option) {}

  bool parse(llvm::cl::Option &option, StringRef argName, StringRef argValue,
             LoweringOptions &value) {
    bool failed = false;
    value.parse(argValue, [&](Twine error) { failed = option.error(error); });
    return failed;
  }
};

struct loweringOptionsOption
    : llvm::cl::opt<LoweringOptions, false, LoweringOptionsParser> {
  loweringOptionsOption(llvm::cl::OptionCategory &cat)
      : llvm::cl::opt<LoweringOptions, false, LoweringOptionsParser>{
            "lowering-options",
            llvm::cl::desc(
                "Style options.  Valid flags include: "
                "noAlwaysComb, exprInEventControl, disallowPackedArrays, "
                "disallowLocalVariables, verifLabels, emittedLineLength=<n>, "
                "maximumNumberOfTermsPerExpression=<n>, "
                "maximumNumberOfTermsInConcat=<n>, explicitBitcast, "
                "maximumNumberOfVariadicOperands=<n>, "
                "emitReplicatedOpsToHeader, "
                "locationInfoStyle={plain,wrapInAtSquareBracket,none}, "
                "disallowPortDeclSharing, printDebugInfo, useOldEmissionMode, "
                "disallowExpressionInliningInPorts"),
            llvm::cl::cat(cat), llvm::cl::value_desc("option")} {
    llvm::errs() << "\n\nConstructed\n\n";
  }
};

} // namespace circt

#endif // CIRCT_SUPPORT_LOWERINGOPTIONSPARSER_H
