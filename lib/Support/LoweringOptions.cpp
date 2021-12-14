//===- LoweringOptions.cpp - CIRCT Lowering Options -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Options for controlling the lowering process. Contains command line
// option definitions and support.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/LoweringOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// LoweringOptions
//===----------------------------------------------------------------------===//

LoweringOptions::LoweringOptions(StringRef options, ErrorHandlerT errorHandler)
    : LoweringOptions() {
  parse(options, errorHandler);
}

LoweringOptions::LoweringOptions(mlir::ModuleOp module) : LoweringOptions() {
  parseFromAttribute(module);
}

void LoweringOptions::parse(StringRef text, ErrorHandlerT errorHandler) {
  while (!text.empty()) {
    // Remove the first option from the text.
    auto split = text.split(",");
    auto option = split.first.trim();
    text = split.second;
    if (option == "") {
      // Empty options are fine.
    } else if (option == "noAlwaysComb") {
      noAlwaysComb = true;
    } else if (option == "exprInEventControl") {
      allowExprInEventControl = true;
    } else if (option == "disallowPackedArrays") {
      disallowPackedArrays = true;
    } else if (option == "disallowLocalVariables") {
      disallowLocalVariables = true;
    } else if (option == "verifLabels") {
      enforceVerifLabels = true;
    } else if (option.startswith("emittedLineLength=")) {
      option = option.drop_front(strlen("emittedLineLength="));
      if (option.getAsInteger(10, emittedLineLength)) {
        errorHandler("expected integer source width");
        emittedLineLength = DEFAULT_LINE_LENGTH;
      }
    } else if (option.startswith("maximumNumberOfTokensPerExpression=")) {
      option = option.drop_front(strlen("maximumNumberOfTokensPerExpression="));
      if (option.getAsInteger(10, maximumNumberOfTokensPerExpression)) {
        errorHandler("expected integer source width");
        maximumNumberOfTokensPerExpression = DEFAULT_TOKEN_NUMBER;
      }
    } else {
      errorHandler(llvm::Twine("unknown style option \'") + option + "\'");
      // We continue parsing options after a failure.
    }
  }
  if (maximumNumberOfTokensPerExpression < emittedLineLength)
    errorHandler("maximumNumberOfTokensPerExpression must be equal or larger "
                 "than emittedLineLength");
}

std::string LoweringOptions::toString() const {
  std::string options = "";
  // All options should add a trailing comma to simplify the code.
  if (noAlwaysComb)
    options += "noAlwaysComb,";
  if (allowExprInEventControl)
    options += "exprInEventControl,";
  if (disallowPackedArrays)
    options += "disallowPackedArrays,";
  if (disallowLocalVariables)
    options += "disallowLocalVariables,";
  if (enforceVerifLabels)
    options += "verifLabels,";

  if (emittedLineLength != DEFAULT_LINE_LENGTH)
    options += "emittedLineLength=" + std::to_string(emittedLineLength) + ',';
  if (maximumNumberOfTokensPerExpression != DEFAULT_TOKEN_NUMBER)
    options += "maximumNumberOfTokensPerExpression=" +
               std::to_string(maximumNumberOfTokensPerExpression) + ',';

  // Remove a trailing comma if present.
  if (!options.empty()) {
    assert(options.back() == ',' && "all options should add a trailing comma");
    options.pop_back();
  }
  return options;
}

StringAttr LoweringOptions::getAttributeFrom(ModuleOp module) {
  return module->getAttrOfType<StringAttr>("circt.loweringOptions");
}

void LoweringOptions::setAsAttribute(ModuleOp module) {
  module->setAttr("circt.loweringOptions",
                  StringAttr::get(module.getContext(), toString()));
}

void LoweringOptions::parseFromAttribute(ModuleOp module) {
  if (auto styleAttr = getAttributeFrom(module))
    parse(styleAttr.getValue(), [&](Twine error) { module.emitError(error); });
}

//===----------------------------------------------------------------------===//
// Command Line Option Processing
//===----------------------------------------------------------------------===//

namespace {
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

/// Commandline arguments for verilog emission.  Used to dynamically register
/// the command line arguments in multiple tools.
struct LoweringCLOptions {
  llvm::cl::opt<LoweringOptions, false, LoweringOptionsParser> loweringOptions{
      "lowering-options",
      llvm::cl::desc(
          "Style options.  Valid flags include: alwaysFF, "
          "noAlwaysComb, exprInEventControl, disallowPackedArrays, "
          "disallowLocalVariables, verifLabels, emittedLineLength=<n>, "
          "maximumNumberOfTokensPerExpression=<n>"),
      llvm::cl::value_desc("option")};
};
} // namespace

/// The staticly initialized command line options.
static llvm::ManagedStatic<LoweringCLOptions> clOptions;

void circt::registerLoweringCLOptions() { *clOptions; }

void circt::applyLoweringCLOptions(ModuleOp module) {
  // If the command line options were not registered in the first place, there
  // is nothing to parse.
  if (!clOptions.isConstructed())
    return;

  // If an output style is applied on the command line, all previous options are
  // discarded.
  if (clOptions->loweringOptions.getNumOccurrences()) {
    clOptions->loweringOptions.setAsAttribute(module);
  }
}
