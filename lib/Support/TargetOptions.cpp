//===- TargetOptions.cpp - CIRCT Lowering Options -------------------------===//
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

#include "circt/Support/TargetOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringExtras.h"

using namespace circt;
using namespace mlir;
using namespace llvm;

//===----------------------------------------------------------------------===//
// TargetOptions
//===----------------------------------------------------------------------===//

class TargetOptionParser {
public:
  TargetOptionParser(MLIRContext *ctx, StringRef options,
                     TargetOptions::ErrorHandlerT errorHandler,
                     TargetOptions::Options &res)
      : ctx(ctx), optionStr(options), errorHandler(errorHandler), res(res) {}

  LogicalResult parse();

private:
  LogicalResult parseKeyword(StringRef text) {
    return success(optionStr.consume_front(text));
  }

  LogicalResult parseAlphanum(std::string *dst);

  LogicalResult parseOption(TargetOptions::Options &dst);
  LogicalResult parseOptions(TargetOptions::Options &res);

  MLIRContext *ctx;
  llvm::StringRef optionStr;
  TargetOptions::ErrorHandlerT errorHandler;
  TargetOptions::Options &res;
};

LogicalResult TargetOptionParser::parseAlphanum(std::string *dst) {
  while (true) {
    if (optionStr.empty())
      return failure();

    StringRef c = optionStr.take_front(1);
    if (!isAlnum(c.front()))
      break;

    (*dst) += c;
    optionStr = optionStr.drop_front(1);
  }
  return success();
}

LogicalResult TargetOptionParser::parseOption(TargetOptions::Options &dst) {
  TargetOptions::Option option;
  // An option can be either a standalone keyword, or a keyword followed
  // by an assignment and a string or number. If a standalone keyword is
  // present, that is a true/false flag.
  // For either case, there may be a trailing ( ... ) where internally there
  // may recursively be more options.

  // Parse the keyword by popping the option string until a non-alphanumeric
  // character is found. optionStr.take_front(1) can be used.
  std::string keyword;
  if (failed(parseAlphanum(&keyword)))
    return failure();

  auto keywordAttr = StringAttr::get(ctx, keyword);
  if (dst.contains(keywordAttr))
    return failure();

  // Is this an assigned option?
  if (succeeded(parseKeyword("="))) {
    // Parse the value - just support integer and strings for now.
    std::string valueStr;
    if (failed(parseAlphanum(&valueStr)))
      return failure();

    int64_t intValue;
    if (to_integer(valueStr, intValue))
      option.value = IntegerAttr::get(IntegerType::get(ctx, 64), intValue);
    else
      option.value = StringAttr::get(ctx, valueStr);
  } else {
    // Flag option
    option.value = UnitAttr::get(ctx);
  }

  // Parse optional nested option-of-options
  if (succeeded(parseKeyword("("))) {
    option.options = TargetOptions::Options();
    if (failed(parseOptions(*option.options)))
      return failure();

    if (failed(parseKeyword(")")))
      return failure();
  }

  dst[keywordAttr] = option;
  return success();
}

LogicalResult TargetOptionParser::parseOptions(TargetOptions::Options &dest) {
  // Parse the options
  while (true) {
    if (failed(parseOption(dest)))
      return failure();

    if (failed(parseKeyword("-")))
      break;
  }

  return success();
}

LogicalResult TargetOptionParser::parse() {
  // Preceding quote
  if (failed(parseKeyword("\'")))
    return failure();

  // Parse the option(s)
  if (failed(parseOptions(res)))
    return failure();

  // Trailing quote
  if (failed(parseKeyword("\'")))
    return failure();

  return success();
}

// Target options are on the form:
// 'language=sv(standard=1995(invalid),no-systemverilog-struct)),target=fpga(family="7-series",model="..."),vendor=mentor'
void TargetOptions::parse(StringRef text, ErrorHandlerT errorHandler) {
  TargetOptionParser parser(ctx, text, errorHandler, options);
  if (failed(parser.parse()))
    errorHandler("failed to parse target options");
}
