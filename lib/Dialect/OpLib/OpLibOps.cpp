//===- OpLibOps.cpp - OpLib Dialect Operations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the OpLib ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OpLib/OpLibOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/OpLib/OpLibAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace circt;
using namespace circt::oplib;

//===----------------------------------------------------------------------===//
// LibraryOp
//===----------------------------------------------------------------------===//

LogicalResult LibraryOp::verify() {
  for (auto &op : this->getBodyRegion().getOps()) {
    if (!isa<OperatorOp>(op))
      return emitOpError("can only contain OperatorOps");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// OperatorOp
//===----------------------------------------------------------------------===//

LogicalResult OperatorOp::verify() {
  auto *term = getBodyRegion().front().getTerminator();
  if (!isa<CalyxMatchOp>(term)) {
    return emitOpError("region terminator must be supported match op");
  }

  if (getIncDelay().has_value() != getOutDelay().has_value()) {
    return emitOpError(
        "must have either both incDelay and outDelay or neither");
  }

  if (getLatency() == 0 &&
      (getIncDelay().has_value() || getOutDelay().has_value())) {
    if (getIncDelay() != getOutDelay()) {
      return emitError(
          "incDelay and outDelay of combinational operators must be the same");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TargetOp
//===----------------------------------------------------------------------===//

ParseResult TargetOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void TargetOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// CalyxMatchOp
//===----------------------------------------------------------------------===//

LogicalResult
CalyxMatchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  StringRef targetName = getTarget();
  Type type = getTargetType();

  // Try to find the referenced target.
  auto fn = symbolTable.lookupSymbolIn<TargetOp>(
      this->getParentOp<OperatorOp>(),
      StringAttr::get(getContext(), targetName));
  if (!fn)
    return emitOpError() << "reference to undefined target '" << targetName
                         << "'";

  // Check that the referenced function has the correct type.
  if (fn.getFunctionType() != type)
    return emitOpError("reference to target with mismatched type");

  return success();
}

//===----------------------------------------------------------------------===//
// Wrappers for the `custom<Properties>` ODS directive.
//===----------------------------------------------------------------------===//

static ParseResult parseOpLibProperties(OpAsmParser &parser, ArrayAttr &attr) {
  auto result = parseOptionalPropertyArray(attr, parser);
  if (!result.has_value() || succeeded(*result))
    return success();
  return failure();
}

static void printOpLibProperties(OpAsmPrinter &p, Operation *op,
                                 ArrayAttr attr) {
  if (!attr)
    return;
  printPropertyArray(attr, p);
}

static ParseResult parseTargetTypes(OpAsmParser &parser, TypeAttr &attr) {
  auto &builder = parser.getBuilder();
  SmallVector<OpAsmParser::Argument> arguments;

  auto result = parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        // Parse argument name if present.
        OpAsmParser::Argument argument;
        auto argPresent = parser.parseOptionalArgument(
            argument, /*allowType=*/true, /*allowAttrs=*/true);
        if (argPresent.has_value()) {
          if (failed(argPresent.value()))
            return failure(); // Present but malformed.

          // Reject this if the preceding argument was missing a name.
          if (!arguments.empty() && arguments.back().ssaName.name.empty())
            return parser.emitError(argument.ssaName.location,
                                    "expected type instead of SSA identifier");

        } else {
          return failure();
        }
        arguments.push_back(argument);
        return success();
      });
  if (failed(result))
    return failure();

  // Parse the function signature.
  SMLoc signatureLocation = parser.getCurrentLocation();
  std::string errorMessage;
  SmallVector<Type> argTypes;
  SmallVector<Type> resultTypes;
  argTypes.reserve(arguments.size());
  for (auto &arg : arguments)
    argTypes.push_back(arg.type);
  Type type = builder.getFunctionType(argTypes, resultTypes);
  if (!type) {
    return parser.emitError(signatureLocation)
           << "failed to construct function type"
           << (errorMessage.empty() ? "" : ": ") << errorMessage;
  }
  attr = TypeAttr::get(type);

  return success();
}

static void printTargetTypes(OpAsmPrinter &p, Operation *op, TypeAttr attr) {
  if (!attr)
    return;

  Region &body = op->getRegion(1);

  auto funcType = llvm::dyn_cast_or_null<FunctionType>(attr.getValue());

  if (funcType == nullptr)
    return;

  ArrayRef<Type> argTypes = funcType.getInputs();
  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    ArrayRef<NamedAttribute> attrs;
    p.printRegionArgument(body.getArgument(i), attrs);
  }

  p << ')';
}

//===----------------------------------------------------------------------===//
// TableGen'ed code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/OpLib/OpLib.cpp.inc"
