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
    return program.emitOpError("must contain one component named "
                               "\"main\" as the entry point.");
  return success();
}

//===----------------------------------------------------------------------===//
// ComponentOp
//===----------------------------------------------------------------------===//

/// Gets the WiresOp of a component.
static WiresOp getWiresOp(ComponentOp componentOp) {
  WiresOp wiresOp;
  auto body = componentOp.getBody();
  auto opIt = llvm::find_if(*body, [](auto &&op) { return isa<WiresOp>(op); });

  assert(opIt != body->end() && "A component should have a WiresOp.");
  return dyn_cast<WiresOp>(*opIt);
}

/// Returns the type of the given component as a function type.
static FunctionType getComponentType(ComponentOp component) {
  return component.getTypeAttr().getValue().cast<FunctionType>();
}

/// Returns the component port names in the given direction.
static ArrayAttr getComponentPortNames(ComponentOp component,
                                       PortDirection direction) {

  if (direction == PortDirection::INPUT)
    return component.inPortNames();
  return component.outPortNames();
}

/// Returns the port information for the given component.
SmallVector<ComponentPortInfo> calyx::getComponentPortInfo(Operation *op) {
  assert(isa<ComponentOp>(op) &&
         "Can only get port information from a component.");
  auto component = dyn_cast<ComponentOp>(op);

  auto functionType = getComponentType(component);
  auto inPortTypes = functionType.getInputs();
  auto outPortTypes = functionType.getResults();
  auto inPortNamesAttr = getComponentPortNames(component, PortDirection::INPUT);
  auto outPortNamesAttr =
      getComponentPortNames(component, PortDirection::OUTPUT);

  SmallVector<ComponentPortInfo> results;
  for (size_t i = 0, e = inPortTypes.size(); i != e; ++i) {
    results.push_back({inPortNamesAttr[i].cast<StringAttr>(), inPortTypes[i],
                       PortDirection::INPUT});
  }
  for (size_t i = 0, e = outPortTypes.size(); i != e; ++i) {
    results.push_back({outPortNamesAttr[i].cast<StringAttr>(), outPortTypes[i],
                       PortDirection::OUTPUT});
  }
  return results;
}

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

  auto functionType = getComponentType(op);
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
  // Verify there is exactly one of each section:
  // calyx.wires, and calyx.control.
  uint32_t numWires = 0, numControl = 0;
  for (auto &bodyOp : *op.getBody()) {
    if (isa<WiresOp>(bodyOp))
      ++numWires;
    else if (isa<ControlOp>(bodyOp))
      ++numControl;
  }
  if (numWires == 1 && numControl == 1)
    return success();

  return op.emitOpError() << "requires exactly one of each: "
                             "'calyx.wires', 'calyx.control'.";
}

//===----------------------------------------------------------------------===//
// CellOp
//===----------------------------------------------------------------------===//

/// Lookup the component for the symbol. This returns null on
/// invalid IR.
ComponentOp CellOp::getReferencedComponent() {
  auto program = (*this)->getParentOfType<ProgramOp>();
  if (!program)
    return nullptr;

  return program.lookupSymbol<ComponentOp>(componentName());
}

static LogicalResult verifyCellOp(CellOp cell) {
  if (cell.componentName() == "main")
    return cell.emitOpError("cannot reference the entry point.");

  // Verify the referenced component exists in this program.
  ComponentOp referencedComponent = cell.getReferencedComponent();
  if (!referencedComponent)
    return cell.emitOpError()
           << "is referencing component: " << cell.componentName()
           << ", which does not exist.";

  // Verify the referenced component is not instantiating itself.
  auto parentComponent = cell->getParentOfType<ComponentOp>();
  if (parentComponent == referencedComponent)
    return cell.emitOpError()
           << "is a recursive instantiation of its parent component: "
           << cell.componentName();

  // Verify the instance result ports with those of its referenced component.
  SmallVector<ComponentPortInfo> componentPorts =
      getComponentPortInfo(referencedComponent);

  size_t numResults = cell.getNumResults();
  if (numResults != componentPorts.size())
    return cell.emitOpError()
           << "has a wrong number of results; expected: "
           << componentPorts.size() << " but got " << numResults;

  for (size_t i = 0; i != numResults; ++i) {
    auto resultType = cell.getResult(i).getType();
    auto expectedType = componentPorts[i].type;
    if (resultType == expectedType)
      continue;
    return cell.emitOpError()
           << "result type for " << componentPorts[i].name << " must be "
           << expectedType << ", but got " << resultType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyAssignOp(AssignOp assign) {
  auto parent = assign->getParentOp();
  if (!(isa<GroupOp, WiresOp>(parent)))
    return assign.emitOpError(
        "should only be contained in 'calyx.wires' or 'calyx.group'");

  return success();
}

//===----------------------------------------------------------------------===//
// ControlOp
//===----------------------------------------------------------------------===//

/// A helper function to verify that each operation in
/// the body of a control-like operation is valid.
static LogicalResult verifyControlLikeOpBody(Operation *op) {
  assert(op->getNumRegions() != 0 && "The operation should have a region.");
  auto &region = op->getRegion(0);
  assert(region.hasOneBlock() && "The region should have one block.");

  bool isNotControlOp = !isa<ControlOp>(op);
  for (auto &&bodyOp : region.front()) {
    if (isa<SeqOp>(bodyOp))
      continue;

    if (isNotControlOp && isa<EnableOp>(bodyOp))
      // An EnableOp may be nested in any control-like
      // operation except "calyx.control". This is verified
      // in the ODS of EnableOp, but kept here for correctness.
      continue;

    return op->emitOpError()
           << "has operation: " << bodyOp.getName()
           << ", which is not allowed in this control-like operation";
  }
  return success();
}

static LogicalResult verifyControlOp(ControlOp controlOp) {
  return verifyControlLikeOpBody(controlOp);
}

//===----------------------------------------------------------------------===//
// SeqOp
//===----------------------------------------------------------------------===//
static LogicalResult verifySeqOp(SeqOp seqOp) {
  return verifyControlLikeOpBody(seqOp);
}

//===----------------------------------------------------------------------===//
// EnableOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyEnableOp(EnableOp enableOp) {
  auto component = enableOp->getParentOfType<ComponentOp>();
  auto wiresOp = getWiresOp(component);
  auto groupName = enableOp.groupName();

  if (!wiresOp.lookupSymbol<GroupOp>(groupName))
    return enableOp.emitOpError()
           << "with group: " << groupName << ", which does not exist.";

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Calyx/Calyx.cpp.inc"
