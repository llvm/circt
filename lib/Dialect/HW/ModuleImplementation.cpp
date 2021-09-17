//===- ModuleImplementation.cpp - Utilities for module-like ops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace mlir;
using namespace circt::hw;

/// Get the portname from an SSA value string, if said value name is not a
/// number
StringAttr module_like_impl::getPortNameAttr(MLIRContext *context,
                                             StringRef name) {
  if (!name.empty()) {
    // Ignore numeric names like %42
    assert(name.size() > 1 && name[0] == '%' && "Unknown MLIR name");
    if (isdigit(name[1]))
      name = StringRef();
    else
      name = name.drop_front();
  }
  return StringAttr::get(context, name);
}

/// Parse a portname as a keyword or a quote surrounded string, followed by a
/// colon.
StringAttr module_like_impl::parsePortName(OpAsmParser &parser) {
  StringAttr result;
  StringRef keyword;
  if (succeeded(parser.parseOptionalKeyword(&keyword))) {
    result = parser.getBuilder().getStringAttr(keyword);
  } else if (parser.parseAttribute(result,
                                   parser.getBuilder().getType<NoneType>()))
    return {};
  return succeeded(parser.parseColon()) ? result : StringAttr();
}

/// Parse a function result list.
///
///   function-result-list ::= function-result-list-parens
///   function-result-list-parens ::= `(` `)`
///                                 | `(` function-result-list-no-parens `)`
///   function-result-list-no-parens ::= function-result (`,` function-result)*
///   function-result ::= (percent-identifier `:`) type attribute-dict?
///
ParseResult module_like_impl::parseFunctionResultList(
    OpAsmParser &parser, SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<NamedAttrList> &resultAttrs,
    SmallVectorImpl<Attribute> &resultNames) {
  if (parser.parseLParen())
    return failure();

  // Special case for an empty set of parens.
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  // Parse individual function results.
  do {
    resultNames.push_back(parsePortName(parser));
    if (!resultNames.back())
      return failure();

    resultTypes.emplace_back();
    resultAttrs.emplace_back();
    if (parser.parseType(resultTypes.back()) ||
        parser.parseOptionalAttrDict(resultAttrs.back()))
      return failure();
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRParen();
}

/// This is a variant of mlor::parseFunctionSignature that allows names on
/// result arguments.
ParseResult module_like_impl::parseModuleFunctionSignature(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::OperandType> &argNames,
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<NamedAttrList> &argAttrs,
    bool &isVariadic, SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<NamedAttrList> &resultAttrs,
    SmallVectorImpl<Attribute> &resultNames) {

  using namespace mlir::function_like_impl;
  bool allowArgAttrs = true;
  bool allowVariadic = false;
  if (parseFunctionArgumentList(parser, allowArgAttrs, allowVariadic, argNames,
                                argTypes, argAttrs, isVariadic))
    return failure();

  if (succeeded(parser.parseOptionalArrow()))
    return parseFunctionResultList(parser, resultTypes, resultAttrs,
                                   resultNames);
  return success();
}

/// Return true if this string parses as a valid MLIR keyword, false otherwise.
bool circt::hw::module_like_impl::isValidKeyword(StringRef name) {
  if (name.empty() || (!isalpha(name[0]) && name[0] != '_'))
    return false;
  for (auto c : name.drop_front()) {
    if (!isalpha(c) && !isdigit(c) && c != '_' && c != '$' && c != '.')
      return false;
  }

  return true;
}

void circt::hw::module_like_impl::printModuleSignature(
    OpAsmPrinter &p, Operation *op, ArrayRef<Type> argTypes, bool isVariadic,
    ArrayRef<Type> resultTypes, bool &needArgNamesAttr) {
  using namespace mlir::function_like_impl;

  Region &body = op->getRegion(0);
  bool isExternal = body.empty();
  SmallString<32> resultNameStr;

  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    auto argName = getModuleArgumentName(op, i);

    if (!isExternal) {
      // Get the printed format for the argument name.
      resultNameStr.clear();
      llvm::raw_svector_ostream tmpStream(resultNameStr);
      p.printOperand(body.front().getArgument(i), tmpStream);

      // If the name wasn't printable in a way that agreed with argName, make
      // sure to print out an explicit argNames attribute.
      if (tmpStream.str().drop_front() != argName)
        needArgNamesAttr = true;

      p << tmpStream.str() << ": ";
    } else if (!argName.empty()) {
      p << '%' << argName << ": ";
    }

    p.printType(argTypes[i]);
    p.printOptionalAttrDict(getArgAttrs(op, i));
  }

  if (isVariadic) {
    if (!argTypes.empty())
      p << ", ";
    p << "...";
  }

  p << ')';

  // We print result types specially since we support named arguments.
  if (!resultTypes.empty()) {
    p << " -> (";
    for (size_t i = 0, e = resultTypes.size(); i < e; ++i) {
      if (i != 0)
        p << ", ";
      StringAttr name = getModuleResultNameAttr(op, i);
      if (isValidKeyword(name.getValue()))
        p << name.getValue();
      else
        p << name;
      p << ": ";

      p.printType(resultTypes[i]);
      p.printOptionalAttrDict(getResultAttrs(op, i));
    }
    p << ')';
  }
}
