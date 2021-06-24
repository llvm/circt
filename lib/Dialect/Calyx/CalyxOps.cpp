//===- CalyxOps.cpp - Calyx op code defs ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is where op definitions live.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::calyx;
using namespace mlir;

//===----------------------------------------------------------------------===//
// ProgramOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyProgramOp(ProgramOp program) {
  if (!program.getMainComponent())
    return program.emitOpError("Must contain one component named "
                               "\"main\" as the entry point.");
  return success();
}

//===----------------------------------------------------------------------===//
// ComponentOp
//===----------------------------------------------------------------------===//

/// Prints the port definitions of a Calyx component signature.
static void printPortDefList(OpAsmPrinter &p, ArrayRef<Type> portDefTypes,
                             ArrayAttr portDefNames) {
  p << '(';
  llvm::interleaveComma(
      llvm::zip(portDefNames, portDefTypes), p, [&](auto nameAndType) {
        if (auto name =
                std::get<0>(nameAndType).template dyn_cast<StringAttr>()) {
          p << name.getValue() << ": ";
        }
        p << std::get<1>(nameAndType);
      });
  p << ')';
}

static void printComponentOp(OpAsmPrinter &p, ComponentOp &op) {
  auto componentName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << "calyx.component ";
  p.printSymbolName(componentName);

  auto typeAttr = op->getAttrOfType<TypeAttr>(ComponentOp::getTypeAttrName());
  auto functionType = typeAttr.getValue().cast<FunctionType>();

  auto inputPortTypes = functionType.getInputs();
  auto inputPortNames = op->getAttrOfType<ArrayAttr>("inPortNames");
  printPortDefList(p, inputPortTypes, inputPortNames);
  p << " -> ";
  auto outputPortTypes = functionType.getResults();
  auto outputPortNames = op->getAttrOfType<ArrayAttr>("outPortNames");
  printPortDefList(p, outputPortTypes, outputPortNames);

  p.printRegion(op.body(), /*printBlockTerminators=*/false,
                /*printEmptyBlock=*/false);
}

/// Parses the ports of a Calyx component signature, and adds the corresponding
/// port names to `attrName`.
static ParseResult
parsePortDefList(OpAsmParser &parser, MLIRContext *context,
                 OperationState &result,
                 SmallVectorImpl<OpAsmParser::OperandType> &ports,
                 SmallVectorImpl<Type> &portTypes, StringRef attrName) {
  if (parser.parseLParen())
    return failure();

  do {
    OpAsmParser::OperandType port;
    Type portType;
    if (failed(parser.parseOptionalRegionArgument(port)) ||
        failed(parser.parseOptionalColon()) ||
        failed(parser.parseType(portType)))
      continue;
    ports.push_back(port);
    portTypes.push_back(portType);
  } while (succeeded(parser.parseOptionalComma()));

  // Add attribute for port names; these are currently
  // just inferred from the arguments of the component.
  SmallVector<Attribute> portNames(ports.size());
  llvm::transform(ports, portNames.begin(), [&](auto port) -> StringAttr {
    return StringAttr::get(context, port.name);
  });
  result.addAttribute(attrName, ArrayAttr::get(context, portNames));

  return (parser.parseRParen());
}

/// Parses the signature of a Calyx component.
static ParseResult
parseComponentSignature(OpAsmParser &parser, OperationState &result,
                        SmallVectorImpl<OpAsmParser::OperandType> &inPorts,
                        SmallVectorImpl<Type> &inPortTypes,
                        SmallVectorImpl<OpAsmParser::OperandType> &outPorts,
                        SmallVectorImpl<Type> &outPortTypes) {
  auto *context = parser.getBuilder().getContext();
  if (parsePortDefList(parser, context, result, inPorts, inPortTypes,
                       "inPortNames") ||
      parser.parseArrow() ||
      parsePortDefList(parser, context, result, outPorts, outPortTypes,
                       "outPortNames"))
    return failure();

  return success();
}

static ParseResult parseComponentOp(OpAsmParser &parser,
                                    OperationState &result) {
  using namespace mlir::function_like_impl;

  StringAttr componentName;
  if (parser.parseSymbolName(componentName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::OperandType> inPorts, outPorts;
  SmallVector<Type> inPortTypes, outPortTypes;
  if (parseComponentSignature(parser, result, inPorts, inPortTypes, outPorts,
                              outPortTypes))
    return failure();

  // Build the component's type for FunctionLike trait.
  auto &builder = parser.getBuilder();
  auto type = builder.getFunctionType(inPortTypes, outPortTypes);
  result.addAttribute(ComponentOp::getTypeAttrName(), TypeAttr::get(type));

  // The entry block needs to have same number of
  // input port definitions as the component.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, inPorts, inPortTypes))
    return failure();

  if (body->empty())
    body->push_back(new Block());

  return success();
}

static LogicalResult verifyComponentOp(ComponentOp op) {
  auto program = op->getParentOfType<ProgramOp>();
  if (!program)
    return op.emitOpError("Should be embedded in 'calyx.program'.");

  // Verify there is exactly one of each section:
  // calyx.cells, calyx.wires, and calyx.control.
  uint32_t numCells = 0, numWires = 0, numControl = 0;
  for (auto &bodyOp : *op.getBody()) {
    if (isa<CellsOp>(bodyOp))
      ++numCells;
    else if (isa<WiresOp>(bodyOp))
      ++numWires;
    else if (isa<ControlOp>(bodyOp))
      ++numControl;
  }
  if (numCells == 1 && numWires == 1 && numControl == 1)
    return success();

  return op.emitOpError()
         << "Requires exactly one of each: "
            "\"calyx.cells\", \"calyx.wires\", \"calyx.control\".";
}

//===----------------------------------------------------------------------===//
// CellsOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyCellsOp(CellsOp cells) {
  auto component = cells->getParentOfType<ComponentOp>();
  if (!component)
    return cells.emitOpError("Should be embedded in 'calyx.component'.");

  for (auto &op : *cells.getBody()) {
    if (!isa<CellOp>(op))
      return cells.emitOpError("Should only contain 'calyx.cell' instances.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CellOp
//===----------------------------------------------------------------------===//

/// Lookup the component for the symbol. This returns null on
/// invalid IR.
Operation *CellOp::getReferencedComponent() {
  auto program = (*this)->getParentOfType<ProgramOp>();
  if (!program)
    return nullptr;

  return program.lookupSymbol(componentName());
}

static LogicalResult verifyCellOp(CellOp cell) {
  // Verify the cell is within the "calyx.cells" sub-section.
  auto cells = cell->getParentOfType<CellsOp>();
  if (!cells)
    return cell.emitOpError("Should be embedded in 'calyx.cells'.");

  // Verify the referenced component exists in this program.
  auto referencedComponent = cell.getReferencedComponent();
  if (!referencedComponent)
    return cell.emitOpError()
           << "Referenced component: " << cell.componentName()
           << " does not exist.";

  // Verify the referenced component is not instantiating itself.
  auto parentComponent = cells->getParentOfType<ComponentOp>();
  if (parentComponent == referencedComponent)
    return cell.emitOpError()
           << "Recursive instantiation of its parent component: "
           << cell.componentName();

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Calyx/Calyx.cpp.inc"
