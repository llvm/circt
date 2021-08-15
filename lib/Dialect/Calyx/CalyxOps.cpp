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

LogicalResult calyx::verifyCell(Operation *op) {
  auto opParent = op->getParentOp();
  if (!isa<ComponentOp>(opParent))
    return op->emitOpError()
           << "has parent: " << opParent << ". This should be ComponentOp.";
  return success();
}

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
/// time with respect to the number of instances within the component.
template <typename Op>
static Op getControlOrWiresFrom(ComponentOp op) {
  auto body = op.getBody();
  // We verify there is a single WiresOp and ControlOp,
  // so this is safe.
  auto opIt = body->getOps<Op>().begin();
  return *opIt;
}

/// Returns the Block argument with the given name from a ComponentOp.
/// If the name doesn't exist, returns an empty Value.
static Value getBlockArgumentWithName(StringRef name, ComponentOp op) {
  ArrayAttr portNames = op.portNames();

  for (size_t i = 0, e = portNames.size(); i != e; ++i) {
    auto portName = portNames[i].cast<StringAttr>();
    if (portName.getValue() == name)
      return op.getBody()->getArgument(i);
  }
  return Value{};
}

} // namespace

WiresOp calyx::ComponentOp::getWiresOp() {
  return getControlOrWiresFrom<WiresOp>(*this);
}

ControlOp calyx::ComponentOp::getControlOp() {
  return getControlOrWiresFrom<ControlOp>(*this);
}

Value calyx::ComponentOp::getGoPort() {
  return getBlockArgumentWithName("go", *this);
}

Value calyx::ComponentOp::getDonePort() {
  return getBlockArgumentWithName("done", *this);
}

/// Returns the type of the given component as a function type.
static FunctionType getComponentType(ComponentOp component) {
  return component.getTypeAttr().getValue().cast<FunctionType>();
}

/// Returns the port information for the given component.
SmallVector<ComponentPortInfo> calyx::getComponentPortInfo(Operation *op) {
  assert(isa<ComponentOp>(op) &&
         "Can only get port information from a component.");
  auto component = dyn_cast<ComponentOp>(op);
  auto portTypes = getComponentType(component).getInputs();
  auto portNamesAttr = component.portNames();
  uint64_t numInPorts = component.numInPorts();

  SmallVector<ComponentPortInfo> results;
  for (uint64_t i = 0, e = portNamesAttr.size(); i != e; ++i) {
    auto dir = i < numInPorts ? PortDirection::INPUT : PortDirection::OUTPUT;
    results.push_back({portNamesAttr[i].cast<StringAttr>(), portTypes[i], dir});
  }
  return results;
}

static void printComponentOp(OpAsmPrinter &p, ComponentOp &op) {
  auto componentName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << "calyx.component ";
  p.printSymbolName(componentName);

  auto ports = getComponentPortInfo(op);
  SmallVector<ComponentPortInfo, 8> inPorts, outPorts;
  for (auto &&port : ports) {
    if (port.direction == PortDirection::INPUT)
      inPorts.push_back(port);
    else
      outPorts.push_back(port);
  }

  auto printPortDefList = [&](auto ports) {
    p << "(";
    llvm::interleaveComma(ports, p, [&](auto port) {
      p << "%" << port.name.getValue() << ": " << port.type;
    });
    p << ")";
  };
  printPortDefList(inPorts);
  p << " -> ";
  printPortDefList(outPorts);

  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false,
                /*printEmptyBlock=*/false);
}

/// Parses the ports of a Calyx component signature, and adds the corresponding
/// port names to `attrName`.
static ParseResult
parsePortDefList(OpAsmParser &parser, OperationState &result,
                 SmallVectorImpl<OpAsmParser::OperandType> &ports,
                 SmallVectorImpl<Type> &portTypes) {
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

  return parser.parseRParen();
}

/// Parses the signature of a Calyx component.
static ParseResult
parseComponentSignature(OpAsmParser &parser, OperationState &result,
                        SmallVectorImpl<OpAsmParser::OperandType> &ports,
                        SmallVectorImpl<Type> &portTypes) {
  if (parsePortDefList(parser, result, ports, portTypes))
    return failure();

  // Record the number of input ports.
  size_t numInPorts = ports.size();

  if (parser.parseArrow() || parsePortDefList(parser, result, ports, portTypes))
    return failure();

  auto *context = parser.getBuilder().getContext();
  // Add attribute for port names; these are currently
  // just inferred from the SSA names of the component.
  SmallVector<Attribute> portNames(ports.size());
  llvm::transform(ports, portNames.begin(), [&](auto port) -> StringAttr {
    StringRef name = port.name;
    if (name.startswith("%"))
      name = name.drop_front();
    return StringAttr::get(context, name);
  });
  result.addAttribute("portNames", ArrayAttr::get(context, portNames));

  // Record the number of input ports.
  result.addAttribute("numInPorts",
                      parser.getBuilder().getI64IntegerAttr(numInPorts));

  return success();
}

static ParseResult parseComponentOp(OpAsmParser &parser,
                                    OperationState &result) {
  using namespace mlir::function_like_impl;

  StringAttr componentName;
  if (parser.parseSymbolName(componentName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::OperandType> ports;
  SmallVector<Type> portTypes;
  if (parseComponentSignature(parser, result, ports, portTypes))
    return failure();

  // Build the component's type for FunctionLike trait. All ports are listed as
  // arguments so they may be accessed within the component.
  auto type =
      parser.getBuilder().getFunctionType(portTypes, /*resultTypes=*/{});
  result.addAttribute(ComponentOp::getTypeAttrName(), TypeAttr::get(type));

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, ports, portTypes))
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

  // Verify the number of input ports.
  SmallVector<ComponentPortInfo> componentPorts = getComponentPortInfo(op);
  uint64_t expectedNumInPorts =
      op->getAttrOfType<IntegerAttr>("numInPorts").getInt();
  uint64_t actualNumInPorts = llvm::count_if(componentPorts, [](auto port) {
    return port.direction == PortDirection::INPUT;
  });
  if (expectedNumInPorts != actualNumInPorts)
    return op.emitOpError()
           << "has mismatched number of in ports. Expected: "
           << expectedNumInPorts << ", actual: " << actualNumInPorts;

  // Verify the component has the following ports.
  // TODO(Calyx): Eventually, we want to attach attributes to these arguments.
  bool go = false, clk = false, reset = false, done = false;
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

  SmallVector<Type, 8> portTypes;
  SmallVector<Attribute, 8> portNames;
  uint64_t numInPorts = 0;
  for (auto &&port : ports) {
    if (port.direction == PortDirection::INPUT)
      ++numInPorts;
    portNames.push_back(port.name);
    portTypes.push_back(port.type);
  }

  // Build the function type of the component.
  auto functionType = builder.getFunctionType(portTypes, {});
  result.addAttribute(getTypeAttrName(), TypeAttr::get(functionType));

  // Record the port names and number of input ports of the component.
  result.addAttribute("portNames", builder.getArrayAttr(portNames));
  result.addAttribute("numInPorts", builder.getI64IntegerAttr(numInPorts));

  // Create a single-blocked region.
  result.addRegion();
  Region *regionBody = result.regions[0].get();
  Block *block = new Block();
  regionBody->push_back(block);

  // Add all ports to the body block.
  block->addArguments(portTypes);

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
// InstanceOp
//===----------------------------------------------------------------------===//

/// Lookup the component for the symbol. This returns null on
/// invalid IR.
ComponentOp InstanceOp::getReferencedComponent() {
  auto program = (*this)->getParentOfType<ProgramOp>();
  if (!program)
    return nullptr;

  return program.lookupSymbol<ComponentOp>(componentName());
}

/// Provide meaningful names to the result values of a InstanceOp.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  auto component = getReferencedComponent();
  auto portNames = component.portNames();

  std::string prefix = instanceName().str() + ".";
  for (size_t i = 0, e = portNames.size(); i != e; ++i) {
    StringRef portName = portNames[i].cast<StringAttr>().getValue();
    setNameFn(getResult(i), prefix + portName.str());
  }
}

static LogicalResult verifyInstanceOp(InstanceOp instance) {
  if (instance.componentName() == "main")
    return instance.emitOpError("cannot reference the entry point.");

  // Verify the referenced component exists in this program.
  ComponentOp referencedComponent = instance.getReferencedComponent();
  if (!referencedComponent)
    return instance.emitOpError()
           << "is referencing component: " << instance.componentName()
           << ", which does not exist.";

  // Verify the referenced component is not instantiating itself.
  auto parentComponent = instance->getParentOfType<ComponentOp>();
  if (parentComponent == referencedComponent)
    return instance.emitOpError()
           << "is a recursive instantiation of its parent component: "
           << instance.componentName();

  // Verify the instance result ports with those of its referenced component.
  SmallVector<ComponentPortInfo> componentPorts =
      getComponentPortInfo(referencedComponent);

  size_t numResults = instance.getNumResults();
  if (numResults != componentPorts.size())
    return instance.emitOpError()
           << "has a wrong number of results; expected: "
           << componentPorts.size() << " but got " << numResults;

  for (size_t i = 0; i != numResults; ++i) {
    auto resultType = instance.getResult(i).getType();
    auto expectedType = componentPorts[i].type;
    if (resultType == expectedType)
      continue;
    return instance.emitOpError()
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
