//===- ParsingUtils.cpp - CIRCT parsing common functions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/ParsingUtils.h"

using namespace circt;

ParseResult circt::parsing_util::parseInitializerList(
    OpAsmParser &parser,
    llvm::SmallVector<OpAsmParser::Argument> &inputArguments,
    llvm::SmallVector<OpAsmParser::UnresolvedOperand> &inputOperands,
    llvm::SmallVector<Type> &inputTypes, ArrayAttr &inputNames) {
  llvm::SmallVector<Attribute> names;
  if (failed(parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
            OpAsmParser::UnresolvedOperand inputOperand;
            Type type;
            auto &arg = inputArguments.emplace_back();
            if (parser.parseArgument(arg) || parser.parseColonType(type) ||
                parser.parseEqual() || parser.parseOperand(inputOperand))
              return failure();

            inputOperands.push_back(inputOperand);
            inputTypes.push_back(type);
            arg.type = type;
            names.push_back(StringAttr::get(
                parser.getContext(),
                /*drop leading %*/ arg.ssaName.name.drop_front()));
            return success();
          })))
    return failure();

  inputNames = ArrayAttr::get(parser.getContext(), names);
  return success();
}

void circt::parsing_util::printInitializerList(OpAsmPrinter &p, ValueRange ins,
                                               ArrayRef<BlockArgument> args) {
  p << "(";
  llvm::interleaveComma(llvm::zip(ins, args), p, [&](auto it) {
    auto [in, arg] = it;
    p << arg << " : " << in.getType() << " = " << in;
  });
  p << ")";
}
