//===- SimOps.cpp - Implement the Sim operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements `sim` dialect ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace circt;
using namespace sim;

ParseResult DPIFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto builder = parser.getBuilder();
  // Parse visibility.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<hw::module_like_impl::PortParse> ports;
  TypeAttr modType;
  if (failed(
          hw::module_like_impl::parseModuleSignature(parser, ports, modType)))
    return failure();

  result.addAttribute(DPIFuncOp::getModuleTypeAttrName(result.name), modType);

  // Convert the specified array of dictionary attrs (which may have null
  // entries) to an ArrayAttr of dictionaries.
  auto unknownLoc = builder.getUnknownLoc();
  SmallVector<Attribute> attrs, locs;
  auto nonEmptyLocsFn = [unknownLoc](Attribute attr) {
    return attr && cast<Location>(attr) != unknownLoc;
  };

  for (auto &port : ports) {
    attrs.push_back(port.attrs ? port.attrs : builder.getDictionaryAttr({}));
    locs.push_back(port.sourceLoc ? Location(*port.sourceLoc) : unknownLoc);
  }

  result.addAttribute(DPIFuncOp::getPerArgumentAttrsAttrName(result.name),
                      builder.getArrayAttr(attrs));
  result.addRegion();

  if (llvm::any_of(locs, nonEmptyLocsFn))
    result.addAttribute(DPIFuncOp::getArgumentLocsAttrName(result.name),
                        builder.getArrayAttr(locs));

  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  return success();
}

LogicalResult
sim::DPICallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto referencedOp = dyn_cast_or_null<sim::DPIFuncOp>(
      symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr()));
  if (!referencedOp)
    return emitError("cannot find function declaration '")
           << getCallee() << "'";
  return success();
}

void DPIFuncOp::print(OpAsmPrinter &p) {
  DPIFuncOp op = *this;
  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';
  p.printSymbolName(funcName);
  hw::module_like_impl::printModuleSignatureNew(
      p, op->getRegion(0), op.getModuleType(),
      getPerArgumentAttrsAttr()
          ? ArrayRef<Attribute>(getPerArgumentAttrsAttr().getValue())
          : ArrayRef<Attribute>{},
      getArgumentLocs() ? SmallVector<Location>(
                              getArgumentLocs().value().getAsRange<Location>())
                        : ArrayRef<Location>{});

  mlir::function_interface_impl::printFunctionAttributes(
      p, op,
      {visibilityAttrName, getModuleTypeAttrName(),
       getPerArgumentAttrsAttrName(), getArgumentLocsAttrName()});
}

OpFoldResult FormatLitOp::fold(FoldAdaptor adaptor) { return getLiteralAttr(); }

OpFoldResult FormatDecOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == IntegerType::get(getContext(), 0U))
    return StringAttr::get(getContext(), "0");

  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getValue())) {
    SmallVector<char, 16> strBuf;
    intAttr.getValue().toString(strBuf, 10U, getIsSigned());

    unsigned width = intAttr.getType().getIntOrFloatBitWidth();
    unsigned padWidth = FormatDecOp::getDecimalWidth(width, getIsSigned());
    padWidth = padWidth > strBuf.size() ? padWidth - strBuf.size() : 0;

    SmallVector<char, 8> padding(padWidth, ' ');
    return StringAttr::get(getContext(), Twine(padding) + Twine(strBuf));
  }
  return {};
}

OpFoldResult FormatHexOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == IntegerType::get(getContext(), 0U))
    return StringAttr::get(getContext(), "");

  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getValue())) {
    SmallVector<char, 8> strBuf;
    intAttr.getValue().toString(strBuf, 16U, /*Signed*/ false,
                                /*formatAsCLiteral*/ false,
                                /*UpperCase*/ false);

    unsigned width = intAttr.getType().getIntOrFloatBitWidth();
    unsigned padWidth = width / 4;
    if (width % 4 != 0)
      padWidth++;
    padWidth = padWidth > strBuf.size() ? padWidth - strBuf.size() : 0;

    SmallVector<char, 8> padding(padWidth, '0');
    return StringAttr::get(getContext(), Twine(padding) + Twine(strBuf));
  }
  return {};
}

OpFoldResult FormatBinOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == IntegerType::get(getContext(), 0U))
    return StringAttr::get(getContext(), "");

  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getValue())) {
    SmallVector<char, 32> strBuf;
    intAttr.getValue().toString(strBuf, 2U, false);

    unsigned width = intAttr.getType().getIntOrFloatBitWidth();
    unsigned padWidth = width > strBuf.size() ? width - strBuf.size() : 0;

    SmallVector<char, 32> padding(padWidth, '0');
    return StringAttr::get(getContext(), Twine(padding) + Twine(strBuf));
  }
  return {};
}

OpFoldResult FormatCharOp::fold(FoldAdaptor adaptor) {
  auto width = getValue().getType().getIntOrFloatBitWidth();
  if (width > 8)
    return {};
  if (width == 0)
    return StringAttr::get(getContext(), Twine(static_cast<char>(0)));

  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getValue())) {
    auto intValue = intAttr.getValue().getZExtValue();
    return StringAttr::get(getContext(), Twine(static_cast<char>(intValue)));
  }

  return {};
}

static StringAttr concatLiterals(MLIRContext *ctxt, ArrayRef<StringRef> lits) {
  assert(!lits.empty() && "No literals to concatenate");
  if (lits.size() == 1)
    return StringAttr::get(ctxt, lits.front());
  SmallString<64> newLit;
  for (auto lit : lits)
    newLit += lit;
  return StringAttr::get(ctxt, newLit);
}

OpFoldResult FormatStringConcatOp::fold(FoldAdaptor adaptor) {
  if (getNumOperands() == 0)
    return StringAttr::get(getContext(), "");
  if (getNumOperands() == 1)
    return getOperand(0);

  // Fold if all operands are literals.
  SmallVector<StringRef> lits;
  for (auto attr : adaptor.getInputs()) {
    auto lit = dyn_cast_or_null<StringAttr>(attr);
    if (!lit)
      return {};
    lits.push_back(lit);
  }
  return concatLiterals(getContext(), lits);
}

LogicalResult FormatStringConcatOp::verify() {
  if (llvm::any_of(getOperands(),
                   [&](Value operand) { return operand == getResult(); }))
    return emitOpError("is infinitely recursive.");
  return success();
}

LogicalResult FormatStringConcatOp::canonicalize(FormatStringConcatOp op,
                                                 PatternRewriter &rewriter) {
  if (op.getNumOperands() < 2)
    return failure(); // Should be handled by the folder

  auto fmtStrType = FormatStringType::get(op.getContext());

  // Check if there are adjacent literals we can merge or empty literals to
  // remove
  SmallVector<StringRef> litSequence;
  SmallVector<Value> newOperands;
  newOperands.reserve(op.getNumOperands());
  FormatLitOp prevLitOp;

  for (auto operand : op.getOperands()) {
    if (auto litOp = operand.getDefiningOp<FormatLitOp>()) {
      if (!litOp.getLiteral().empty()) {
        prevLitOp = litOp;
        litSequence.push_back(litOp.getLiteral());
      }
    } else {
      if (!litSequence.empty()) {
        if (litSequence.size() > 1) {
          // Create a fused literal.
          auto newLit = rewriter.createOrFold<FormatLitOp>(
              op.getLoc(), fmtStrType,
              concatLiterals(op.getContext(), litSequence));
          newOperands.push_back(newLit);
        } else {
          // Reuse the existing literal.
          newOperands.push_back(prevLitOp.getResult());
        }
        litSequence.clear();
      }
      newOperands.push_back(operand);
    }
  }

  // Push trailing literals into the new operand list
  if (!litSequence.empty()) {
    if (litSequence.size() > 1) {
      // Create a fused literal.
      auto newLit = rewriter.createOrFold<FormatLitOp>(
          op.getLoc(), fmtStrType,
          concatLiterals(op.getContext(), litSequence));
      newOperands.push_back(newLit);
    } else {
      // Reuse the existing literal.
      newOperands.push_back(prevLitOp.getResult());
    }
  }

  if (newOperands.size() == op.getNumOperands())
    return failure(); // Nothing changed

  if (newOperands.empty())
    rewriter.replaceOpWithNewOp<FormatLitOp>(op, fmtStrType,
                                             rewriter.getStringAttr(""));
  else if (newOperands.size() == 1)
    rewriter.replaceOp(op, newOperands);
  else
    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(newOperands); });

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Sim/Sim.cpp.inc"
