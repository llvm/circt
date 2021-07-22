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
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::calyx;
using namespace mlir;

LogicalResult calyx::verifyControlLikeOp(Operation *op) {
  auto parent = op->getParentOp();
  // Operations that may parent other ControlLike operations.
  auto isValidParent = [](Operation *operation) {
    return isa<ControlOp, SeqOp>(operation);
  };
  if (!isValidParent(parent))
    return op->emitOpError()
           << "has parent: " << parent
           << ", which is not allowed for a control-like operation.";

  if (op->getNumRegions() == 0)
    return success();

  auto &region = op->getRegion(0);
  // Operations that are allowed in the body of a ControlLike op.
  auto isValidBodyOp = [](Operation *operation) {
    return isa<EnableOp, SeqOp>(operation);
  };
  for (auto &&bodyOp : region.front()) {
    if (isValidBodyOp(&bodyOp))
      continue;

    return op->emitOpError()
           << "has operation: " << bodyOp.getName()
           << ", which is not allowed in this control-like operation";
  }
  return success();
}

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

namespace {

/// This is a helper function that should only be used to get the WiresOp or
/// ControlOp of a ComponentOp, which are guaranteed to exist and generally at
/// the end of a component's body. In the worst case, this will run in linear
/// time with respect to the number of instances within the cell.
template <typename Op>
static Op getControlOrWiresFrom(ComponentOp op) {
  auto body = op.getBody();
  // We verify there is a single WiresOp and ControlOp,
  // so this is safe.
  auto opIt = body->getOps<Op>().begin();
  return *opIt;
}

} // namespace

WiresOp calyx::ComponentOp::getWiresOp() {
  return getControlOrWiresFrom<WiresOp>(*this);
}

ControlOp calyx::ComponentOp::getControlOp() {
  return getControlOrWiresFrom<ControlOp>(*this);
}

OpResult calyx::ComponentOp::getDonePort() {
  ArrayAttr outPortNames = this->outPortNames();

  size_t i = 0, numArgs = this->getNumFuncResults();
  for (; i != numArgs; ++i) {
    StringRef portName = outPortNames[i].cast<StringAttr>().getValue();
    if (portName == "done")
      break;
  }
  return (*this)->getOpResult(i);
}

Value calyx::ComponentOp::getGoPort() {
  ArrayAttr inPortNames = this->inPortNames();

  size_t i = 0, numArgs = this->getNumFuncArguments();
  for (; i != numArgs; ++i) {
    StringRef portName = inPortNames[i].cast<StringAttr>().getValue();
    if (portName == "go")
      break;
  }
  return (*this).getBody()->getArgument(i);
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
          p << '%' << name.getValue() << ": ";
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
    StringRef name = port.name;
    if (name.startswith("%"))
      name = name.drop_front();
    return StringAttr::get(context, name);
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
  if (!(numWires == 1) || !(numControl == 1))
    return op.emitOpError() << "requires exactly one of each: "
                               "'calyx.wires', 'calyx.control'.";

  // Verify the component has the following ports.
  // TODO(Calyx): Eventually, we want to use either types for these,
  //  e.g. `calyx.clk_type` or attributes for passes.
  bool go = false, clk = false, reset = false, done = false;
  SmallVector<ComponentPortInfo> componentPorts = getComponentPortInfo(op);
  for (auto &&port : componentPorts) {
    if (!port.type.isInteger(1))
      // Each of the ports has bit width 1.
      continue;

    StringRef portName = port.name.getValue();
    if (port.direction == PortDirection::OUTPUT) {
      done |= (portName == "done");
    } else {
      go |= (portName == "go");
      clk |= (portName == "clk");
      reset |= (portName == "reset");
    }
    if (go && clk && reset && done)
      return success();
  }
  return op->emitOpError() << "does not have required 1-bit input ports `go`, "
                              "`clk`, `reset`, and output port `done`";
}

void ComponentOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<ComponentPortInfo> ports) {
  using namespace mlir::function_like_impl;

  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  SmallVector<Type, 4> inPortTypes, outPortTypes;
  SmallVector<Attribute, 4> inPortNames, outPortNames;

  for (auto &&port : ports) {
    if (port.direction == PortDirection::INPUT) {
      inPortTypes.push_back(port.type);
      inPortNames.push_back(port.name);
    } else {
      outPortTypes.push_back(port.type);
      outPortNames.push_back(port.name);
    }
  }

  // Build the function type of the component.
  auto functionType = builder.getFunctionType(inPortTypes, outPortTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(functionType));

  // Record the port names of the component.
  result.addAttribute("inPortNames", builder.getArrayAttr(inPortNames));
  result.addAttribute("outPortNames", builder.getArrayAttr(outPortNames));

  // Create a single-blocked region.
  result.addRegion();
  Region *regionBody = result.regions[0].get();
  Block *block = new Block();
  regionBody->push_back(block);

  // Add input ports to the body block.
  for (auto port : ports) {
    if (port.direction == PortDirection::OUTPUT)
      continue;
    block->addArgument(port.type);
  }

  // Insert the WiresOp and ControlOp.
  IRRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  builder.create<WiresOp>(result.location);
  builder.create<ControlOp>(result.location);
}

//===----------------------------------------------------------------------===//
// ControlOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyControlOp(ControlOp control) {
  auto body = control.getBody();

  // A control operation may have a single EnableOp within it. However,
  // that must be the only operation. E.g.
  // Allowed:      calyx.control { calyx.enable @A }
  // Not Allowed:  calyx.control { calyx.enable @A calyx.seq { ... } }
  if (llvm::any_of(*body, [](auto &&op) { return isa<EnableOp>(op); }) &&
      body->getOperations().size() > 1)
    return control->emitOpError(
        "EnableOp is not a composition operator. It should be nested "
        "in a control flow operation, such as \"calyx.seq\"");

  return success();
}

//===----------------------------------------------------------------------===//
// WiresOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyWiresOp(WiresOp wires) {
  auto component = wires->getParentOfType<ComponentOp>();
  auto control = component.getControlOp();

  // Verify each group is referenced in the control section.
  for (auto &&op : *wires.getBody()) {
    if (!isa<GroupOp>(op))
      continue;
    auto group = cast<GroupOp>(op);
    StringRef groupName = group.sym_name();
    if (SymbolTable::symbolKnownUseEmpty(groupName, control))
      return op.emitOpError() << "with name: " << groupName
                              << " is unused in the control execution schedule";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GroupOp
//===----------------------------------------------------------------------===//
GroupGoOp GroupOp::getGoOp() {
  auto body = this->getBody();
  auto opIt = body->getOps<GroupGoOp>().begin();
  return *opIt;
}

GroupDoneOp GroupOp::getDoneOp() {
  auto body = this->getBody();
  return cast<GroupDoneOp>(body->getTerminator());
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

/// Provide meaningful names to the result values of a CellOp.
void CellOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  auto component = getReferencedComponent();

  std::string prefix = instanceName().str() + ".";
  size_t resultIndex = 0;
  auto renameResults = [&](ArrayAttr portNames) {
    for (size_t i = 0, e = portNames.size(); i != e; ++i, ++resultIndex) {
      StringRef portName = portNames[i].cast<StringAttr>().getValue();
      setNameFn(getResult(resultIndex), prefix + portName.str());
    }
  };
  renameResults(component.inPortNames());
  renameResults(component.outPortNames());
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
// GroupGoOp
//===----------------------------------------------------------------------===//

/// Provide meaningful names to the result value of a GroupGoOp.
void GroupGoOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  auto parent = (*this)->getParentOfType<GroupOp>();
  auto name = parent.sym_name();
  std::string resultName = name.str() + ".go";
  setNameFn(getResult(), resultName);
}

//===----------------------------------------------------------------------===//
// RegisterOp
//===----------------------------------------------------------------------===//

/// Provide meaningful names to the result values of a RegisterOp.
void RegisterOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Provide default names for instance results.
  StringRef registerName = this->name();
  std::string prefix = registerName.str() + ".";

  setNameFn(getResult(0), prefix + "in");
  setNameFn(getResult(1), prefix + "write_en");
  setNameFn(getResult(2), prefix + "clk");
  setNameFn(getResult(3), prefix + "reset");
  setNameFn(getResult(4), prefix + "out");
  setNameFn(getResult(5), prefix + "done");
}

//===----------------------------------------------------------------------===//
// EnableOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyEnableOp(EnableOp enableOp) {
  auto component = enableOp->getParentOfType<ComponentOp>();
  auto wiresOp = component.getWiresOp();
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
