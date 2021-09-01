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
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::calyx;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Utilities related to Direction
//===----------------------------------------------------------------------===//

Direction direction::get(bool a) { return static_cast<Direction>(a); }

SmallVector<Direction> direction::genInOutDirections(size_t nIns,
                                                     size_t nOuts) {
  SmallVector<Direction> dirs;
  std::generate_n(std::back_inserter(dirs), nIns,
                  [] { return Direction::Input; });
  std::generate_n(std::back_inserter(dirs), nOuts,
                  [] { return Direction::Output; });
  return dirs;
}

IntegerAttr direction::packAttribute(ArrayRef<Direction> directions,
                                     MLIRContext *ctx) {
  // Pack the array of directions into an APInt.  Input is zero, output is one.
  size_t numDirections = directions.size();
  APInt portDirections(numDirections, 0);
  for (size_t i = 0, e = numDirections; i != e; ++i)
    if (directions[i] == Direction::Output)
      portDirections.setBit(i);

  return IntegerAttr::get(IntegerType::get(ctx, numDirections), portDirections);
}

/// Turn a packed representation of port attributes into a vector that can be
/// worked with.
SmallVector<Direction> direction::unpackAttribute(Operation *component) {
  APInt value =
      component->getAttr(direction::attrKey).cast<IntegerAttr>().getValue();

  SmallVector<Direction> result;
  auto bitWidth = value.getBitWidth();
  result.reserve(bitWidth);
  for (size_t i = 0, e = bitWidth; i != e; ++i)
    result.push_back(direction::get(value[i]));
  return result;
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Verifies the body of a ControlLikeOp.
static LogicalResult verifyControlBody(Operation *op) {
  if (isa<SeqOp, ParOp>(op))
    // This does not apply to sequential and parallel regions.
    return success();

  // Some ControlLike operations have (possibly) multiple regions, e.g. IfOp.
  for (auto &region : op->getRegions()) {
    auto opsIt = region.getOps();
    size_t numOperations = std::distance(opsIt.begin(), opsIt.end());
    // A body of a ControlLike operation may have a single EnableOp within it.
    // However, that must be the only operation.
    //  E.g. Allowed:  calyx.control { calyx.enable @A }
    //   Not Allowed:  calyx.control { calyx.enable @A calyx.seq { ... } }
    bool isInvalidBody =
        numOperations > 1 && llvm::any_of(region.front(), [](auto &&bodyOp) {
          return isa<EnableOp>(bodyOp);
        });
    if (isInvalidBody)
      return op->emitOpError(
          "EnableOp is not a composition operator. It should be nested "
          "in a control flow operation, such as \"calyx.seq\"");
  }
  return success();
}

/// Checks whether @p port is driven from within @p groupOp.
static LogicalResult isPortDrivenByGroup(Value port, GroupOp groupOp) {
  // Check if the port is driven by an assignOp from within @p groupOp.
  for (auto &use : port.getUses()) {
    if (auto assignOp = dyn_cast<AssignOp>(use.getOwner())) {
      if (assignOp.dest() != port ||
          assignOp->getParentOfType<GroupOp>() != groupOp)
        continue;
      return success();
    }
  }

  // If @p port is an output of an InstanceOp, and if any input port of
  // this InstanceOp is driven within @p groupOp, we'll assume that @p port
  // is sensitive to the driven input port.
  // @TODO: simplify this logic when the calyx.cell interface allows us to more
  // easily access the input and output ports of a component
  if (auto instanceOp = dyn_cast<InstanceOp>(port.getDefiningOp())) {
    auto compOp = instanceOp.getReferencedComponent();
    auto compOpPortInfo = llvm::enumerate(getComponentPortInfo(compOp));

    bool isOutputPort = llvm::any_of(compOpPortInfo, [&](auto portInfo) {
      return port == instanceOp->getResult(portInfo.index()) &&
             portInfo.value().direction == Direction::Output;
    });

    if (isOutputPort) {
      return success(llvm::any_of(compOpPortInfo, [&](auto portInfo) {
        return portInfo.value().direction == Direction::Input &&
               succeeded(isPortDrivenByGroup(
                   instanceOp->getResult(portInfo.index()), groupOp));
      }));
    }
  }

  return failure();
}

LogicalResult calyx::verifyCell(Operation *op) {
  auto opParent = op->getParentOp();
  if (!isa<ComponentOp>(opParent))
    return op->emitOpError()
           << "has parent: " << opParent << ", expected ComponentOp.";
  if (!op->hasAttr("instanceName"))
    return op->emitOpError() << "does not have an instanceName attribute.";

  return success();
}

LogicalResult calyx::verifyControlLikeOp(Operation *op) {
  auto parent = op->getParentOp();
  // Operations that may parent other ControlLike operations.
  auto isValidParent = [](Operation *operation) {
    return isa<ControlOp, SeqOp, IfOp, WhileOp, ParOp>(operation);
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
    return isa<EnableOp, SeqOp, IfOp, WhileOp, ParOp>(operation);
  };
  for (auto &&bodyOp : region.front()) {
    if (isValidBodyOp(&bodyOp))
      continue;

    return op->emitOpError()
           << "has operation: " << bodyOp.getName()
           << ", which is not allowed in this control-like operation";
  }
  return verifyControlBody(op);
}

// Convenience function for getting the SSA name of @p v under the scope of
// operation @p scopeOp
static std::string valueName(Operation *scopeOp, Value v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  AsmState asmState(scopeOp);
  v.printAsOperand(os, asmState);
  return s;
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
  auto portDirectionsAttr =
      component->getAttrOfType<mlir::IntegerAttr>(direction::attrKey);

  SmallVector<ComponentPortInfo> results;
  for (uint64_t i = 0, e = portNamesAttr.size(); i != e; ++i) {
    results.push_back({portNamesAttr[i].cast<StringAttr>(), portTypes[i],
                       direction::get(portDirectionsAttr.getValue()[i])});
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
    if (port.direction == Direction::Input)
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
  SmallVector<OpAsmParser::OperandType> inPorts, outPorts;
  SmallVector<Type> inPortTypes, outPortTypes;

  if (parsePortDefList(parser, result, inPorts, inPortTypes))
    return failure();

  if (parser.parseArrow() ||
      parsePortDefList(parser, result, outPorts, outPortTypes))
    return failure();

  auto *context = parser.getBuilder().getContext();
  // Add attribute for port names; these are currently
  // just inferred from the SSA names of the component.
  SmallVector<Attribute> portNames;
  auto getPortName = [context](const auto &port) -> StringAttr {
    StringRef name = port.name;
    if (name.startswith("%"))
      name = name.drop_front();
    return StringAttr::get(context, name);
  };
  llvm::transform(inPorts, std::back_inserter(portNames), getPortName);
  llvm::transform(outPorts, std::back_inserter(portNames), getPortName);

  result.addAttribute("portNames", ArrayAttr::get(context, portNames));
  result.addAttribute(
      direction::attrKey,
      direction::packAttribute(
          direction::genInOutDirections(inPorts.size(), outPorts.size()),
          context));

  ports.append(inPorts);
  ports.append(outPorts);
  portTypes.append(inPortTypes);
  portTypes.append(outPortTypes);

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

  SmallVector<ComponentPortInfo> componentPorts = getComponentPortInfo(op);

  // Verify the component has the following ports.
  // TODO(Calyx): Eventually, we want to attach attributes to these arguments.
  bool go = false, clk = false, reset = false, done = false;
  for (auto &&port : componentPorts) {
    if (!port.type.isInteger(1))
      // Each of the ports has bit width 1.
      continue;

    StringRef portName = port.name.getValue();
    if (port.direction == Direction::Output) {
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

/// Returns a new vector containing the concatenation of vectors @p a and @p b.
template <typename T>
static SmallVector<T> concat(const SmallVectorImpl<T> &a,
                             const SmallVectorImpl<T> &b) {
  SmallVector<T> out;
  out.append(a);
  out.append(b);
  return out;
}

void ComponentOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<ComponentPortInfo> ports) {
  using namespace mlir::function_like_impl;

  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  std::pair<SmallVector<Type, 8>, SmallVector<Type, 8>> portIOTypes;
  std::pair<SmallVector<Attribute, 8>, SmallVector<Attribute, 8>> portIONames;
  SmallVector<Direction, 8> portDirections;
  // Avoid using llvm::partition or llvm::sort to preserve relative ordering
  // between individual inputs and outputs.
  for (auto &&port : ports) {
    bool isInput = port.direction == Direction::Input;
    (isInput ? portIOTypes.first : portIOTypes.second).push_back(port.type);
    (isInput ? portIONames.first : portIONames.second).push_back(port.name);
  }
  auto portTypes = concat(portIOTypes.first, portIOTypes.second);
  auto portNames = concat(portIONames.first, portIONames.second);

  // Build the function type of the component.
  auto functionType = builder.getFunctionType(portTypes, {});
  result.addAttribute(getTypeAttrName(), TypeAttr::get(functionType));

  // Record the port names and number of input ports of the component.
  result.addAttribute("portNames", builder.getArrayAttr(portNames));
  result.addAttribute(direction::attrKey,
                      direction::packAttribute(direction::genInOutDirections(
                                                   portIOTypes.first.size(),
                                                   portIOTypes.second.size()),
                                               builder.getContext()));

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
  return verifyControlBody(control);
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
// Utilities for operations with the Cell trait.
//===----------------------------------------------------------------------===//

/// Gives each result of the cell a meaningful name in the form:
/// <instance-name>.<port-name>
static void getCellAsmResultNames(OpAsmSetValueNameFn setNameFn, Operation *op,
                                  ArrayRef<StringRef> portNames) {
  assert(op->hasTrait<Cell>() && "must have the Cell trait");

  auto instanceName = op->getAttrOfType<StringAttr>("instanceName").getValue();
  std::string prefix = instanceName.str() + ".";
  for (size_t i = 0, e = portNames.size(); i != e; ++i)
    setNameFn(op->getResult(i), prefix + portNames[i].str());
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

/// Returns whether the given value is a port of a component, which are
/// defined as Block Arguments.
static bool isComponentPort(Value value) { return value.isa<BlockArgument>(); }

/// Verifies the given value of the AssignOp if it is a component port. The
/// `isDestination` boolean is used to distinguish whether the value is a source
/// or a destination.
static LogicalResult verifyAssignOpWithComponentPort(AssignOp op,
                                                     bool isDestination) {
  Value value = isDestination ? op.dest() : op.src();

  auto component = op->getParentOfType<ComponentOp>();
  Block *body = component.getBody();
  auto it = llvm::find_if(body->getArguments(),
                          [&](auto arg) { return arg == value; });
  assert(it != body->getArguments().end() &&
         "Value not found in the component ports.");
  BlockArgument blockArg = *it;

  size_t blockArgNum = blockArg.getArgNumber();

  auto ports = getComponentPortInfo(component);
  assert(blockArgNum < ports.size() &&
         "Block argument index should be within range of component's ports.");
  // Within a component, we want to drive component output ports, and be
  // driven by its input ports.
  Direction validComponentDirection = isDestination ? Output : Input;
  return ports[blockArgNum].direction == validComponentDirection
             ? success()
             : op.emitOpError()
                   << "has a component port as the "
                   << (isDestination ? "destination" : "source")
                   << " with the incorrect direction. It should be "
                   << (isDestination ? "Output" : "Input") << ".";
}

/// Verifies the destination of an assignment operation.
static LogicalResult verifyAssignOpDestination(AssignOp assign) {
  Value dest = assign.dest();
  if (isComponentPort(dest))
    return verifyAssignOpWithComponentPort(assign,
                                           /*isDestination=*/true);

  Operation *definingOp = dest.getDefiningOp();
  bool isValidDestination =
      definingOp->hasTrait<Cell>() || isa<GroupGoOp, GroupDoneOp>(definingOp);
  if (!isValidDestination)
    assign->emitOpError(
        "has an invalid destination port. The destination of an AssignOp "
        "must be drive-able.");

  return success();
}

/// Verifies the source of an assignment operation.
static LogicalResult verifyAssignOpSource(AssignOp assign) {
  Value src = assign.src();
  if (isComponentPort(src))
    return verifyAssignOpWithComponentPort(assign,
                                           /*isDestination=*/false);

  return success();
}

static LogicalResult verifyAssignOp(AssignOp assign) {
  // TODO(Calyx): Verification for Cell trait (inverse of components ports):
  // (1) The destination has Direction::Input, and
  // (2) the source has Direction::Output.
  // This is simpler with the completion of the Cell Interface.
  // See: https://github.com/llvm/circt/issues/1597
  return succeeded(verifyAssignOpDestination(assign)) &&
                 succeeded(verifyAssignOpSource(assign))
             ? success()
             : failure();
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

/// Provide meaningful names to the result values of an InstanceOp.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallVector<StringRef> ports;
  for (auto &&port : getReferencedComponent().portNames())
    ports.push_back(port.cast<StringAttr>().getValue());
  getCellAsmResultNames(setNameFn, *this, ports);
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
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

//===----------------------------------------------------------------------===//
// MemoryOp
//===----------------------------------------------------------------------===//

/// Provide meaningful names to the result values of a MemoryOp.
void MemoryOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallVector<StringRef> portNames;
  SmallVector<SmallString<8>> addrNames;
  for (size_t i = 0, e = addrSizes().size(); i != e; ++i) {
    addrNames.emplace_back("addr" + std::to_string(i));
    portNames.push_back(addrNames[i]);
  }
  portNames.push_back("write_data");
  portNames.push_back("write_en");
  portNames.push_back("read_data");
  portNames.push_back("done");
  getCellAsmResultNames(setNameFn, *this, portNames);
}

void MemoryOp::build(OpBuilder &builder, OperationState &state,
                     Twine instanceName, int64_t width, ArrayRef<int64_t> sizes,
                     ArrayRef<int64_t> addrSizes) {
  state.addAttribute("instanceName", builder.getStringAttr(instanceName));
  state.addAttribute("width", builder.getI64IntegerAttr(width));
  state.addAttribute("sizes", builder.getI64ArrayAttr(sizes));
  state.addAttribute("addrSizes", builder.getI64ArrayAttr(addrSizes));
  SmallVector<Type> types;
  for (int64_t size : addrSizes)
    types.push_back(builder.getIntegerType(size));
  types.push_back(builder.getIntegerType(width));
  types.push_back(builder.getI1Type());
  types.push_back(builder.getIntegerType(width));
  types.push_back(builder.getI1Type());
  state.addTypes(types);
}

static LogicalResult verifyMemoryOp(MemoryOp memoryOp) {
  auto sizes = memoryOp.sizes().getValue();
  auto addrSizes = memoryOp.addrSizes().getValue();
  size_t numDims = memoryOp.sizes().size();
  size_t numAddrs = memoryOp.addrSizes().size();
  if (numDims != numAddrs)
    return memoryOp.emitOpError("mismatched number of dimensions (")
           << numDims << ") and address sizes (" << numAddrs << ")";

  size_t numExtraPorts = 4; // write data/enable and read data/done.
  if (memoryOp.getNumResults() != numAddrs + numExtraPorts)
    return memoryOp.emitOpError("incorrect number of address ports, expected ")
           << numAddrs;

  for (size_t i = 0; i < numDims; ++i) {
    int64_t size = sizes[i].cast<IntegerAttr>().getInt();
    int64_t addrSize = addrSizes[i].cast<IntegerAttr>().getInt();
    if (llvm::Log2_64_Ceil(size) > addrSize)
      return memoryOp.emitOpError("address size (")
             << addrSize << ") for dimension " << i
             << " can't address the entire range (" << size << ")";
  }

  return success();
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
// IfOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyIfOp(IfOp ifOp) {
  auto component = ifOp->getParentOfType<ComponentOp>();
  auto wiresOp = component.getWiresOp();
  auto groupName = ifOp.groupName();
  auto groupOp = wiresOp.lookupSymbol<GroupOp>(groupName);

  if (!groupOp)
    return ifOp.emitOpError()
           << "with group '" << groupName << "', which does not exist.";

  if (ifOp.thenRegion().front().empty())
    return ifOp.emitError() << "empty 'then' region.";

  if (ifOp.elseRegion().getBlocks().size() != 0 &&
      ifOp.elseRegion().front().empty())
    return ifOp.emitError() << "empty 'else' region.";

  if (failed(isPortDrivenByGroup(ifOp.cond(), groupOp)))
    return ifOp.emitError()
           << "conditional op: '" << valueName(component, ifOp.cond())
           << "' expected to be driven from group: '" << ifOp.groupName()
           << "' but no driver was found.";

  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyWhileOp(WhileOp whileOp) {
  auto component = whileOp->getParentOfType<ComponentOp>();
  auto wiresOp = component.getWiresOp();
  auto groupName = whileOp.groupName();
  auto groupOp = wiresOp.lookupSymbol<GroupOp>(groupName);

  if (!groupOp)
    return whileOp.emitOpError()
           << "with group '" << groupName << "', which does not exist.";

  if (whileOp.body().front().empty())
    return whileOp.emitError() << "empty body region.";

  if (failed(isPortDrivenByGroup(whileOp.cond(), groupOp)))
    return whileOp.emitError()
           << "conditional op: '" << valueName(component, whileOp.cond())
           << "' expected to be driven from group: '" << whileOp.groupName()
           << "' but no driver was found.";

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Calyx/Calyx.cpp.inc"
