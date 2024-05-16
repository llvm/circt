//===- MSFTOps.cpp - Implement MSFT dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSFT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// Custom directive parsers/printers
//===----------------------------------------------------------------------===//

static ParseResult parsePhysLoc(OpAsmParser &p, PhysLocationAttr &attr) {
  llvm::SMLoc loc = p.getCurrentLocation();
  StringRef devTypeStr;
  uint64_t x, y, num;

  if (p.parseKeyword(&devTypeStr) || p.parseKeyword("x") || p.parseColon() ||
      p.parseInteger(x) || p.parseKeyword("y") || p.parseColon() ||
      p.parseInteger(y) || p.parseKeyword("n") || p.parseColon() ||
      p.parseInteger(num))
    return failure();

  std::optional<PrimitiveType> devType = symbolizePrimitiveType(devTypeStr);
  if (!devType) {
    p.emitError(loc, "Unknown device type '" + devTypeStr + "'");
    return failure();
  }
  PrimitiveTypeAttr devTypeAttr =
      PrimitiveTypeAttr::get(p.getContext(), *devType);
  attr = PhysLocationAttr::get(p.getContext(), devTypeAttr, x, y, num);
  return success();
}

static void printPhysLoc(OpAsmPrinter &p, Operation *, PhysLocationAttr loc) {
  p << stringifyPrimitiveType(loc.getPrimitiveType().getValue())
    << " x: " << loc.getX() << " y: " << loc.getY() << " n: " << loc.getNum();
}

static ParseResult parseListOptionalRegLocList(OpAsmParser &p,
                                               LocationVectorAttr &locs) {
  SmallVector<PhysLocationAttr, 32> locArr;
  TypeAttr type;
  if (p.parseAttribute(type) || p.parseLSquare() ||
      p.parseCommaSeparatedList(
          [&]() { return parseOptionalRegLoc(locArr, p); }) ||
      p.parseRSquare())
    return failure();

  if (failed(LocationVectorAttr::verify(
          [&p]() { return p.emitError(p.getNameLoc()); }, type, locArr)))
    return failure();
  locs = LocationVectorAttr::get(p.getContext(), type, locArr);
  return success();
}

static void printListOptionalRegLocList(OpAsmPrinter &p, Operation *,
                                        LocationVectorAttr locs) {
  p << locs.getType() << " [";
  llvm::interleaveComma(locs.getLocs(), p, [&p](PhysLocationAttr loc) {
    printOptionalRegLoc(loc, p);
  });
  p << "]";
}

static ParseResult parseImplicitInnerRef(OpAsmParser &p,
                                         hw::InnerRefAttr &innerRef) {
  SymbolRefAttr sym;
  if (p.parseAttribute(sym))
    return failure();
  auto loc = p.getCurrentLocation();
  if (sym.getNestedReferences().size() != 1)
    return p.emitError(loc, "expected <module sym>::<inner name>");
  innerRef = hw::InnerRefAttr::get(
      sym.getRootReference(),
      sym.getNestedReferences().front().getRootReference());
  return success();
}
void printImplicitInnerRef(OpAsmPrinter &p, Operation *,
                           hw::InnerRefAttr innerRef) {
  MLIRContext *ctxt = innerRef.getContext();
  StringRef innerRefNameStr, moduleStr;
  if (innerRef.getTarget())
    innerRefNameStr = innerRef.getTarget().getValue();
  if (innerRef.getRoot())
    moduleStr = innerRef.getRoot().getValue();
  p << SymbolRefAttr::get(ctxt, moduleStr,
                          {FlatSymbolRefAttr::get(ctxt, innerRefNameStr)});
}

//===----------------------------------------------------------------------===//
// DynamicInstanceOp
//===----------------------------------------------------------------------===//

ArrayAttr DynamicInstanceOp::getPath() {
  SmallVector<Attribute, 16> path;
  DynamicInstanceOp next = *this;
  do {
    path.push_back(next.getInstanceRefAttr());
    next = next->getParentOfType<DynamicInstanceOp>();
  } while (next);
  std::reverse(path.begin(), path.end());
  return ArrayAttr::get(getContext(), path);
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

void OutputOp::build(OpBuilder &odsBuilder, OperationState &odsState) {}

//===----------------------------------------------------------------------===//
// MSFT high level design constructs
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SystolicArrayOp
//===----------------------------------------------------------------------===//

ParseResult SystolicArrayOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  uint64_t numRows, numColumns;
  Type rowType, columnType;
  OpAsmParser::UnresolvedOperand rowInputs, columnInputs;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLSquare() || parser.parseOperand(rowInputs) ||
      parser.parseColon() || parser.parseInteger(numRows) ||
      parser.parseKeyword("x") || parser.parseType(rowType) ||
      parser.parseRSquare() || parser.parseLSquare() ||
      parser.parseOperand(columnInputs) || parser.parseColon() ||
      parser.parseInteger(numColumns) || parser.parseKeyword("x") ||
      parser.parseType(columnType) || parser.parseRSquare())
    return failure();

  hw::ArrayType rowInputType = hw::ArrayType::get(rowType, numRows);
  hw::ArrayType columnInputType = hw::ArrayType::get(columnType, numColumns);
  SmallVector<Value> operands;
  if (parser.resolveOperands({rowInputs, columnInputs},
                             {rowInputType, columnInputType}, loc, operands))
    return failure();
  result.addOperands(operands);

  Type peOutputType;
  SmallVector<OpAsmParser::Argument> peArgs;
  if (parser.parseKeyword("pe")) {
    return failure();
  }
  llvm::SMLoc peLoc = parser.getCurrentLocation();
  if (parser.parseArgumentList(peArgs, AsmParser::Delimiter::Paren)) {
    return failure();
  }
  if (peArgs.size() != 2) {
    return parser.emitError(peLoc, "expected two operands");
  }

  peArgs[0].type = rowType;
  peArgs[1].type = columnType;

  if (parser.parseArrow() || parser.parseLParen() ||
      parser.parseType(peOutputType) || parser.parseRParen())
    return failure();

  result.addTypes({hw::ArrayType::get(
      hw::ArrayType::get(peOutputType, numColumns), numRows)});

  Region *pe = result.addRegion();

  peLoc = parser.getCurrentLocation();

  if (parser.parseRegion(*pe, peArgs))
    return failure();

  if (pe->getBlocks().size() != 1)
    return parser.emitError(peLoc, "expected one block for the PE");
  Operation *peTerm = pe->getBlocks().front().getTerminator();
  if (peTerm->getOperands().size() != 1)
    return peTerm->emitOpError("expected one return value");
  if (peTerm->getOperand(0).getType() != peOutputType)
    return peTerm->emitOpError("expected return type as given in parent: ")
           << peOutputType;

  return success();
}

void SystolicArrayOp::print(OpAsmPrinter &p) {
  hw::ArrayType rowInputType = cast<hw::ArrayType>(getRowInputs().getType());
  hw::ArrayType columnInputType = cast<hw::ArrayType>(getColInputs().getType());
  p << " [";
  p.printOperand(getRowInputs());
  p << " : " << rowInputType.getNumElements() << " x ";
  p.printType(rowInputType.getElementType());
  p << "] [";
  p.printOperand(getColInputs());
  p << " : " << columnInputType.getNumElements() << " x ";
  p.printType(columnInputType.getElementType());

  p << "] pe (";
  p.printOperand(getPe().getArgument(0));
  p << ", ";
  p.printOperand(getPe().getArgument(1));
  p << ") -> (";
  p.printType(
      cast<hw::ArrayType>(
          cast<hw::ArrayType>(getPeOutputs().getType()).getElementType())
          .getElementType());
  p << ") ";
  p.printRegion(getPe(), false);
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

LogicalResult LinearOp::verify() {

  for (auto &op : *getBodyBlock()) {
    if (!isa<hw::HWDialect, comb::CombDialect, msft::MSFTDialect>(
            op.getDialect()))
      return emitOpError() << "expected only hw, comb, and msft dialect ops "
                              "inside the datapath.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PDMulticycleOp
//===----------------------------------------------------------------------===//

Operation *PDMulticycleOp::getTopModule(hw::HWSymbolCache &cache) {
  // Both symbols should reference the same top-level module in their respective
  // HierPath ops.
  Operation *srcTop = getHierPathTopModule(getLoc(), cache, getSourceAttr());
  Operation *dstTop = getHierPathTopModule(getLoc(), cache, getDestAttr());
  if (srcTop != dstTop) {
    emitOpError("source and destination paths must refer to the same top-level "
                "module.");
    return nullptr;
  }
  return srcTop;
}

#define GET_OP_CLASSES
#include "circt/Dialect/MSFT/MSFT.cpp.inc"
