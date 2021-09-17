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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
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

/// Verify that the value is not a "complex" value. For example, the source
/// of an AssignOp should be a constant or port, e.g.
/// %and = comb.and %a, %b : i1
/// calyx.assign %port = %and, %c1_i1 ? : i1   // Incorrect
/// calyx.assign %port = %c1_i1, %and ? : i1   // Correct
/// TODO(Calyx): This is useful to verify current MLIR can be lowered to the
/// native compiler. Remove this when Calyx supports wire declarations.
/// See: https://github.com/llvm/circt/pull/1774 for context.
template <typename Op>
static LogicalResult verifyNotComplexSource(Op op) {
  Operation *definingOp = op.src().getDefiningOp();
  if (definingOp == nullptr)
    // This is a port of the parent component.
    return success();

  // Currently, we use the Combinational dialect to perform logical operations
  // on wires, i.e. comb::AndOp, comb::OrOp, comb::XorOp.
  if (auto dialect = definingOp->getDialect(); isa<comb::CombDialect>(dialect))
    return op->emitOpError("has source that is not a port or constant. "
                           "Complex logic should be conducted in the guard.");

  return success();
}

/// Convenience function for getting the SSA name of @p v under the scope of
/// operation @p scopeOp
static std::string valueName(Operation *scopeOp, Value v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  AsmState asmState(scopeOp);
  v.printAsOperand(os, asmState);
  return s;
}

/// Returns whether this value is either (1) a port on a ComponentOp or (2) a
/// port on a cell interface.
static bool isPort(Value value) {
  Operation *definingOp = value.getDefiningOp();
  return value.isa<BlockArgument>() ||
         (definingOp && isa<CellInterface>(definingOp));
}

/// Gets the port for a given BlockArgument.
PortInfo calyx::getPortInfo(BlockArgument arg) {
  Operation *op = arg.getOwner()->getParentOp();
  assert(isa<ComponentOp>(op) &&
         "Only ComponentOp should support lookup by BlockArgument.");
  return cast<ComponentOp>(op).getPortInfo()[arg.getArgNumber()];
}

/// Returns whether the given operation has a control region.
static bool hasControlRegion(Operation *op) {
  return isa<ControlOp, SeqOp, IfOp, WhileOp, ParOp>(op);
}

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
    bool usesEnableAsCompositionOperator =
        numOperations > 1 && llvm::any_of(region.front(), [](auto &&bodyOp) {
          return isa<EnableOp>(bodyOp);
        });
    if (usesEnableAsCompositionOperator)
      return op->emitOpError(
          "EnableOp is not a composition operator. It should be nested "
          "in a control flow operation, such as \"calyx.seq\"");

    // Verify that multiple control flow operations are nested inside a single
    // control operator. See: https://github.com/llvm/circt/issues/1723
    size_t numControlFlowRegions =
        llvm::count_if(opsIt, [](auto &&op) { return hasControlRegion(&op); });
    if (numControlFlowRegions > 1)
      return op->emitOpError(
          "has an invalid control sequence. Multiple control flow operations "
          "must all be nested in a single calyx.seq or calyx.par");
  }
  return success();
}

static LogicalResult anyPortsDrivenByGroup(ValueRange ports,
                                           GroupInterface groupOp);

/// Determines whether the given port is used in the group. Its use depends on
/// the `isDriven` value; if true, then the port should be a destination in an
/// AssignOp. Otherwise, it should be the source, i.e. a read.
static bool portIsUsedInGroup(Value port, GroupInterface group, bool isDriven) {
  return llvm::any_of(port.getUses(), [&](auto &&use) {
    auto assignOp = dyn_cast<AssignOp>(use.getOwner());
    if (assignOp == nullptr)
      return false;

    Operation *parent = assignOp->getParentOp();
    if (isa<WiresOp>(parent))
      // This is a continuous assignment.
      return false;

    // A port is used if it meet the criteria:
    // (1) it is a {source, destination} of an assignment.
    // (2) that assignment is found in the provided group.

    // If not driven, then read.
    Value expected = isDriven ? assignOp.dest() : assignOp.src();
    return expected == port && group == parent;
  });
}

/// Checks whether @p port is driven from within @p groupOp.
static LogicalResult portDrivenByGroup(Value port, GroupInterface groupOp) {
  // Check if the port is driven by an assignOp from within @p groupOp.
  if (portIsUsedInGroup(port, groupOp, /*isDriven=*/true))
    return success();

  // If @p port is an output of a cell then we conservatively enforce that at
  // least one input port of the cell must be driven by the group.
  if (auto cell = dyn_cast<CellInterface>(port.getDefiningOp());
      cell && cell.direction(port) == calyx::Direction::Output)
    return anyPortsDrivenByGroup(cell.getInputPorts(), groupOp);

  return failure();
}

/// Checks whether all ports are driven within the group.
static LogicalResult allPortsDrivenByGroup(ValueRange ports,
                                           GroupInterface group) {
  return success(llvm::all_of(ports, [&](auto port) {
    return portIsUsedInGroup(port, group, /*isDriven=*/true);
  }));
}

/// Checks whether any ports are driven within the group.
static LogicalResult anyPortsDrivenByGroup(ValueRange ports,
                                           GroupInterface group) {
  return success(llvm::any_of(ports, [&](auto port) {
    return portIsUsedInGroup(port, group, /*isDriven=*/true);
  }));
}

/// Checks whether any ports are read within the group.
static LogicalResult anyPortsReadByGroup(ValueRange ports,
                                         GroupInterface group) {
  return success(llvm::any_of(ports, [&](auto port) {
    return portIsUsedInGroup(port, group, /*isDriven=*/false);
  }));
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

/// Verifies that certain ports of primitives are either driven or read
/// together.
static LogicalResult verifyPrimitivePortDriving(AssignOp assign,
                                                GroupInterface group) {
  Operation *destDefiningOp = assign.dest().getDefiningOp();
  if (destDefiningOp == nullptr)
    return success();
  auto destCell = dyn_cast<CellInterface>(destDefiningOp);
  if (destCell == nullptr)
    return success();

  LogicalResult verifyWrites =
      TypeSwitch<Operation *, LogicalResult>(destCell)
          .Case<RegisterOp>([&](auto op) {
            // We only want to verify this is written to if the {write enable,
            // in} port is driven.
            return succeeded(anyPortsDrivenByGroup(
                       {op.writeEnPort(), op.inPort()}, group))
                       ? allPortsDrivenByGroup({op.writeEnPort(), op.inPort()},
                                               group)
                       : success();
          })
          .Case<MemoryOp>([&](auto op) {
            SmallVector<Value> requiredWritePorts;
            // If writing to memory, write_en, write_data, and all address ports
            // should be driven.
            requiredWritePorts.push_back(op.writeEn());
            requiredWritePorts.push_back(op.writeData());
            for (Value address : op.addrPorts())
              requiredWritePorts.push_back(address);

            // We only want to verify the write ports if either write_data or
            // write_en is driven.
            return succeeded(anyPortsDrivenByGroup(
                       {op.writeData(), op.writeEn()}, group))
                       ? allPortsDrivenByGroup(requiredWritePorts, group)
                       : success();
          })
          .Case<AndLibOp, OrLibOp, XorLibOp, AddLibOp, SubLibOp, GtLibOp,
                LtLibOp, EqLibOp, NeqLibOp, GeLibOp, LeLibOp, LshLibOp,
                RshLibOp, SgtLibOp, SltLibOp, SeqLibOp, SneqLibOp, SgeLibOp,
                SleLibOp, SrshLibOp>([&](auto op) {
            Value lhs = op.lhsPort(), rhs = op.rhsPort();
            return succeeded(anyPortsDrivenByGroup({lhs, rhs}, group))
                       ? allPortsDrivenByGroup({lhs, rhs}, group)
                       : success();
          })
          .Default([&](auto op) { return success(); });

  if (failed(verifyWrites))
    return group->emitOpError()
           << "with cell: " << destCell->getName() << " \""
           << destCell.instanceName()
           << "\" is performing a write and failed to drive all necessary "
              "ports.";

  Operation *srcDefiningOp = assign.src().getDefiningOp();
  if (srcDefiningOp == nullptr)
    return success();
  auto srcCell = dyn_cast<CellInterface>(srcDefiningOp);
  if (srcCell == nullptr)
    return success();

  LogicalResult verifyReads =
      TypeSwitch<Operation *, LogicalResult>(srcCell)
          .Case<MemoryOp>([&](auto op) {
            // If reading memory, all address ports should be driven. Note that
            // we only want to verify the read ports if read_data is used in the
            // group.
            return succeeded(anyPortsReadByGroup({op.readData()}, group))
                       ? allPortsDrivenByGroup(op.addrPorts(), group)
                       : success();
          })
          .Default([&](auto op) { return success(); });

  if (failed(verifyReads))
    return group->emitOpError() << "with cell: " << srcCell->getName() << " \""
                                << srcCell.instanceName()
                                << "\" is having a read performed upon it, and "
                                   "failed to drive all necessary ports.";

  return success();
}

LogicalResult calyx::verifyGroupInterface(Operation *op) {
  auto group = dyn_cast<GroupInterface>(op);
  if (group == nullptr)
    return success();

  for (auto &&groupOp : *group.getBody()) {
    auto assign = dyn_cast<AssignOp>(groupOp);
    if (assign == nullptr)
      continue;
    if (failed(verifyPrimitivePortDriving(assign, group)))
      return failure();
  }

  return success();
}

LogicalResult calyx::verifyControlLikeOp(Operation *op) {
  auto parent = op->getParentOp();
  if (!hasControlRegion(parent))
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

SmallVector<PortInfo> ComponentOp::getPortInfo() {
  auto portTypes = getComponentType(*this).getInputs();
  auto portNamesAttr = portNames();
  auto portDirectionsAttr =
      (*this)->getAttrOfType<IntegerAttr>(direction::attrKey);
  auto portAttrs = (*this)->getAttrOfType<ArrayAttr>("portAttributes");

  SmallVector<PortInfo> results;
  for (uint64_t i = 0, e = portNamesAttr.size(); i != e; ++i) {
    results.push_back(PortInfo{portNamesAttr[i].cast<StringAttr>(),
                               portTypes[i],
                               direction::get(portDirectionsAttr.getValue()[i]),
                               portAttrs[i].cast<DictionaryAttr>()});
  }
  return results;
}

/// A helper function to return a filtered subset of a component's ports.
template <typename Pred>
static SmallVector<PortInfo> getFilteredPorts(ComponentOp op, Pred p) {
  SmallVector<PortInfo> ports = op.getPortInfo();
  llvm::erase_if(ports, p);
  return ports;
}

SmallVector<PortInfo> ComponentOp::getInputPortInfo() {
  return getFilteredPorts(
      *this, [](const PortInfo &port) { return port.direction == Output; });
}

SmallVector<PortInfo> ComponentOp::getOutputPortInfo() {
  return getFilteredPorts(
      *this, [](const PortInfo &port) { return port.direction == Input; });
}

static void printComponentOp(OpAsmPrinter &p, ComponentOp op) {
  auto componentName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << " ";
  p.printSymbolName(componentName);

  // Print the port definition list for input and output ports.
  auto printPortDefList = [&](auto ports) {
    p << "(";
    llvm::interleaveComma(ports, p, [&](const PortInfo &port) {
      p << "%" << port.name.getValue() << ": " << port.type;
      if (!port.attributes.empty()) {
        p << " ";
        p.printAttributeWithoutType(port.attributes);
      }
    });
    p << ")";
  };
  printPortDefList(op.getInputPortInfo());
  p << " -> ";
  printPortDefList(op.getOutputPortInfo());

  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false,
                /*printEmptyBlock=*/false);
}

/// Parses the ports of a Calyx component signature, and adds the corresponding
/// port names to `attrName`.
static ParseResult
parsePortDefList(OpAsmParser &parser, OperationState &result,
                 SmallVectorImpl<OpAsmParser::OperandType> &ports,
                 SmallVectorImpl<Type> &portTypes,
                 SmallVectorImpl<NamedAttrList> &portAttrs) {
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

    NamedAttrList portAttr;
    portAttrs.push_back(succeeded(parser.parseOptionalAttrDict(portAttr))
                            ? portAttr
                            : NamedAttrList());

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
  SmallVector<NamedAttrList> portAttributes;

  if (parsePortDefList(parser, result, inPorts, inPortTypes, portAttributes))
    return failure();

  if (parser.parseArrow() ||
      parsePortDefList(parser, result, outPorts, outPortTypes, portAttributes))
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

  SmallVector<Attribute> portAttrs;
  llvm::transform(portAttributes, std::back_inserter(portAttrs),
                  [&](auto attr) { return attr.getDictionary(context); });
  result.addAttribute("portAttributes", ArrayAttr::get(context, portAttrs));

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

  // Build the component's type for FunctionLike trait. All ports are listed
  // as arguments so they may be accessed within the component.
  auto type =
      parser.getBuilder().getFunctionType(portTypes, /*resultTypes=*/{});
  result.addAttribute(ComponentOp::getTypeAttrName(), TypeAttr::get(type));

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, ports, portTypes))
    return failure();

  if (body->empty())
    body->push_back(new Block());

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

/// Determines whether the given ComponentOp has all the required ports.
static LogicalResult hasRequiredPorts(ComponentOp op) {
  // Get all identifiers from the component ports.
  llvm::SmallVector<StringRef, 4> identifiers;
  for (PortInfo &port : op.getPortInfo()) {
    auto portIds = port.getAllIdentifiers();
    identifiers.append(portIds.begin(), portIds.end());
  }
  // Sort the identifiers: a pre-condition for std::set_intersection.
  std::sort(identifiers.begin(), identifiers.end());

  llvm::SmallVector<StringRef, 4> intersection,
      interfacePorts{"clk", "done", "go", "reset"};
  // Find the intersection between all identifiers and required ports.
  std::set_intersection(interfacePorts.begin(), interfacePorts.end(),
                        identifiers.begin(), identifiers.end(),
                        std::back_inserter(intersection));

  if (intersection.size() == interfacePorts.size())
    return success();

  SmallVector<StringRef, 4> difference;
  std::set_difference(interfacePorts.begin(), interfacePorts.end(),
                      intersection.begin(), intersection.end(),
                      std::back_inserter(difference));
  return op->emitOpError()
         << "is missing the following required port attribute identifiers: "
         << difference;
}

static LogicalResult verifyComponentOp(ComponentOp op) {
  // Verify there is exactly one of each the wires and control operations.
  auto wIt = op.getBody()->getOps<WiresOp>();
  auto cIt = op.getBody()->getOps<ControlOp>();
  if (std::distance(wIt.begin(), wIt.end()) +
          std::distance(cIt.begin(), cIt.end()) !=
      2)
    return op.emitOpError(
        "requires exactly one of each: 'calyx.wires', 'calyx.control'.");

  if (failed(hasRequiredPorts(op)))
    return failure();

  // Verify the component actually does something: has a non-empty Control
  // region, or continuous assignments.
  bool hasNoControlConstructs =
      op.getControlOp().getBody()->getOperations().empty();
  bool hasNoAssignments = op.getWiresOp().getBody()->getOps<AssignOp>().empty();
  if (hasNoControlConstructs && hasNoAssignments)
    return op->emitOpError(
        "The component currently does nothing. It needs to either have "
        "continuous assignments in the Wires region or control constructs in "
        "the Control region.");

  return success();
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
                        StringAttr name, ArrayRef<PortInfo> ports) {
  using namespace mlir::function_like_impl;

  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  std::pair<SmallVector<Type, 8>, SmallVector<Type, 8>> portIOTypes;
  std::pair<SmallVector<Attribute, 8>, SmallVector<Attribute, 8>> portIONames;
  std::pair<SmallVector<Attribute, 8>, SmallVector<Attribute, 8>>
      portIOAttributes;
  SmallVector<Direction, 8> portDirections;
  // Avoid using llvm::partition or llvm::sort to preserve relative ordering
  // between individual inputs and outputs.
  for (auto &&port : ports) {
    bool isInput = port.direction == Direction::Input;
    (isInput ? portIOTypes.first : portIOTypes.second).push_back(port.type);
    (isInput ? portIONames.first : portIONames.second).push_back(port.name);
    (isInput ? portIOAttributes.first : portIOAttributes.second)
        .push_back(port.attributes);
  }
  auto portTypes = concat(portIOTypes.first, portIOTypes.second);
  auto portNames = concat(portIONames.first, portIONames.second);
  auto portAttributes = concat(portIOAttributes.first, portIOAttributes.second);

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
  // Record the attributes of the ports.
  result.addAttribute("portAttributes", builder.getArrayAttr(portAttributes));

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
    if (!isa<GroupInterface>(op))
      continue;
    auto group = cast<GroupInterface>(op);
    auto groupName = group.symName();
    if (SymbolTable::symbolKnownUseEmpty(groupName, control))
      return op.emitOpError() << "with name: " << groupName
                              << " is unused in the control execution schedule";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CombGroupOp
//===----------------------------------------------------------------------===//

/// Verifies the defining operation of a value is combinational.
static LogicalResult isCombinational(Value value, GroupInterface group) {
  Operation *definingOp = value.getDefiningOp();
  if (definingOp == nullptr || definingOp->hasTrait<Combinational>())
    // This is a port of the parent component or combinational.
    return success();

  // For now, assumes all component instances are combinational. Once
  // combinational components are supported, this can be strictly enforced.
  if (isa<InstanceOp>(definingOp))
    return success();

  // Constants and logical operations are OK.
  if (isa<comb::CombDialect, hw::HWDialect>(definingOp->getDialect()))
    return success();

  // Reads to MemoryOp and RegisterOp are combinational. Writes are not.
  if (auto r = dyn_cast<RegisterOp>(definingOp)) {
    return value == r.outPort()
               ? success()
               : group->emitOpError()
                     << "with register: \"" << r.instanceName()
                     << "\" is conducting a memory store. This is not "
                        "combinational.";
  } else if (auto m = dyn_cast<MemoryOp>(definingOp)) {
    auto writePorts = {m.writeData(), m.writeEn()};
    return (llvm::none_of(writePorts, [&](Value p) { return p == value; }))
               ? success()
               : group->emitOpError()
                     << "with memory: \"" << m.instanceName()
                     << "\" is conducting a memory store. This "
                        "is not combinational.";
  }

  std::string portName =
      valueName(group->getParentOfType<ComponentOp>(), value);
  return group->emitOpError() << "with port: " << portName
                              << ". This operation is not combinational.";
}

/// Verifies a combinational group may contain only combinational primitives or
/// perform combinational logic.
static LogicalResult verifyCombGroupOp(CombGroupOp group) {

  for (auto &&op : *group.getBody()) {
    auto assign = dyn_cast<AssignOp>(op);
    if (assign == nullptr)
      continue;
    Value dst = assign.dest(), src = assign.src();
    if (failed(isCombinational(dst, group)) ||
        failed(isCombinational(src, group)))
      return failure();
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
  assert(isa<CellInterface>(op) && "must implement the Cell interface");

  auto instanceName = op->getAttrOfType<StringAttr>("instanceName").getValue();
  std::string prefix = instanceName.str() + ".";
  for (size_t i = 0, e = portNames.size(); i != e; ++i)
    setNameFn(op->getResult(i), prefix + portNames[i].str());
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

/// Determines whether the given direction is valid with the given inputs. The
/// `isDestination` boolean is used to distinguish whether the value is a source
/// or a destination.
static LogicalResult verifyPortDirection(AssignOp op, Value value,
                                         bool isDestination) {
  Operation *definingOp = value.getDefiningOp();
  bool isComponentPort = value.isa<BlockArgument>(),
       isCellInterfacePort = definingOp && isa<CellInterface>(definingOp);
  assert((isComponentPort || isCellInterfacePort) && "Not a port.");

  PortInfo port = isComponentPort
                      ? getPortInfo(value.cast<BlockArgument>())
                      : cast<CellInterface>(definingOp).portInfo(value);

  bool isSource = !isDestination;
  // Component output ports and cell interface input ports should be driven.
  Direction validDirection =
      (isDestination && isComponentPort) || (isSource && isCellInterfacePort)
          ? Direction::Output
          : Direction::Input;

  return port.direction == validDirection
             ? success()
             : op.emitOpError()
                   << "has a " << (isComponentPort ? "component" : "cell")
                   << " port as the "
                   << (isDestination ? "destination" : "source")
                   << " with the incorrect direction.";
}

/// Verifies the value of a given assignment operation. The boolean
/// `isDestination` is used to distinguish whether the destination
/// or source of the AssignOp is to be verified.
static LogicalResult verifyAssignOpValue(AssignOp op, bool isDestination) {
  bool isSource = !isDestination;
  Value value = isDestination ? op.dest() : op.src();
  if (isPort(value))
    return verifyPortDirection(op, value, isDestination);

  // A destination may also be the Go or Done hole of a GroupOp.
  if (isDestination && !isa<GroupGoOp, GroupDoneOp>(value.getDefiningOp()))
    return op->emitOpError(
        "has an invalid destination port. It must be drive-able.");
  else if (isSource)
    return verifyNotComplexSource(op);

  return success();
}

static LogicalResult verifyAssignOp(AssignOp assign) {
  bool isDestination = true, isSource = false;
  if (failed(verifyAssignOpValue(assign, isDestination)))
    return failure();
  if (failed(verifyAssignOpValue(assign, isSource)))
    return failure();

  return success();
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

/// Verifies the port information in comparison with the referenced component
/// of an instance. This helper function avoids conducting a lookup for the
/// referenced component twice.
static LogicalResult verifyInstanceOpType(InstanceOp instance,
                                          ComponentOp referencedComponent) {
  if (instance.componentName() == "main")
    return instance.emitOpError("cannot reference the entry point.");

  // Verify the instance result ports with those of its referenced component.
  SmallVector<PortInfo> componentPorts = referencedComponent.getPortInfo();
  size_t numPorts = componentPorts.size();

  size_t numResults = instance.getNumResults();
  if (numResults != numPorts)
    return instance.emitOpError()
           << "has a wrong number of results; expected: " << numPorts
           << " but got " << numResults;

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

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *referencedComponent =
      symbolTable.lookupNearestSymbolFrom(*this, componentNameAttr());
  if (referencedComponent == nullptr)
    return emitError() << "referencing component: " << componentName()
                       << ", which does not exist.";

  // Verify the referenced component is not instantiating itself.
  auto parentComponent = (*this)->getParentOfType<ComponentOp>();
  if (parentComponent == referencedComponent)
    return emitError() << "recursive instantiation of its parent component: "
                       << componentName();

  assert(isa<ComponentOp>(referencedComponent) && "Should be a ComponentOp.");
  return verifyInstanceOpType(*this, cast<ComponentOp>(referencedComponent));
}

/// Provide meaningful names to the result values of an InstanceOp.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

SmallVector<StringRef> InstanceOp::portNames() {
  SmallVector<StringRef> portNames;
  for (auto &&port : getReferencedComponent().portNames())
    portNames.push_back(port.cast<StringAttr>().getValue());
  return portNames;
}

SmallVector<Direction> InstanceOp::portDirections() {
  SmallVector<Direction> portDirections;
  for (const PortInfo &port : getReferencedComponent().getPortInfo())
    portDirections.push_back(port.direction);
  return portDirections;
}

SmallVector<DictionaryAttr> InstanceOp::portAttributes() {
  SmallVector<DictionaryAttr> portAttributes;
  for (const PortInfo &port : getReferencedComponent().getPortInfo())
    portAttributes.push_back(port.attributes);
  return portAttributes;
}

//===----------------------------------------------------------------------===//
// GroupGoOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyGroupGoOp(GroupGoOp goOp) {
  return verifyNotComplexSource(goOp);
}

/// Provide meaningful names to the result value of a GroupGoOp.
void GroupGoOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  auto parent = (*this)->getParentOfType<GroupOp>();
  auto name = parent.sym_name();
  std::string resultName = name.str() + ".go";
  setNameFn(getResult(), resultName);
}

//===----------------------------------------------------------------------===//
// GroupDoneOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyGroupDoneOp(GroupDoneOp doneOp) {
  Operation *srcOp = doneOp.src().getDefiningOp();
  Value optionalGuard = doneOp.guard();
  Operation *guardOp = optionalGuard ? optionalGuard.getDefiningOp() : nullptr;
  bool noGuard = (guardOp == nullptr);

  if (srcOp == nullptr)
    // This is a port of the parent component.
    return success();

  if (isa<hw::ConstantOp>(srcOp) && (noGuard || isa<hw::ConstantOp>(guardOp)))
    return doneOp->emitOpError()
           << "with constant source" << (noGuard ? "" : " and constant guard")
           << ". This should be a combinational group.";

  return verifyNotComplexSource(doneOp);
}

//===----------------------------------------------------------------------===//
// RegisterOp
//===----------------------------------------------------------------------===//

/// Provide meaningful names to the result values of a RegisterOp.
void RegisterOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

SmallVector<StringRef> RegisterOp::portNames() {
  return {"in", "write_en", "clk", "reset", "out", "done"};
}

SmallVector<Direction> RegisterOp::portDirections() {
  return {Input, Input, Input, Input, Output, Output};
}

SmallVector<DictionaryAttr> RegisterOp::portAttributes() {
  MLIRContext *context = getContext();
  IntegerAttr isSet = IntegerAttr::get(IntegerType::get(context, 1), 1);
  NamedAttrList writeEn, clk, reset, done;
  writeEn.append("go", isSet);
  clk.append("clk", isSet);
  reset.append("reset", isSet);
  done.append("done", isSet);
  return {
      DictionaryAttr(),               // In
      writeEn.getDictionary(context), // Write enable
      clk.getDictionary(context),     // Clk
      reset.getDictionary(context),   // Reset
      DictionaryAttr(),               // Out
      done.getDictionary(context)     // Done
  };
}

//===----------------------------------------------------------------------===//
// MemoryOp
//===----------------------------------------------------------------------===//

/// Provide meaningful names to the result values of a MemoryOp.
void MemoryOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

SmallVector<StringRef> MemoryOp::portNames() {
  SmallVector<StringRef> portNames;
  for (size_t i = 0, e = addrSizes().size(); i != e; ++i) {
    auto nameAttr =
        StringAttr::get(this->getContext(), "addr" + std::to_string(i));
    portNames.push_back(nameAttr.getValue());
  }
  portNames.append({"write_data", "write_en", "clk", "read_data", "done"});
  return portNames;
}

SmallVector<Direction> MemoryOp::portDirections() {
  SmallVector<Direction> portDirections;
  for (size_t i = 0, e = addrSizes().size(); i != e; ++i)
    portDirections.push_back(Input);
  portDirections.append({Input, Input, Input, Output, Output});
  return portDirections;
}

SmallVector<DictionaryAttr> MemoryOp::portAttributes() {
  SmallVector<DictionaryAttr> portAttributes;
  MLIRContext *context = getContext();
  for (size_t i = 0, e = addrSizes().size(); i != e; ++i)
    portAttributes.push_back(DictionaryAttr()); // Addresses

  // Use a boolean to indicate this attribute is used.
  IntegerAttr isSet = IntegerAttr::get(IntegerType::get(context, 1), 1);
  NamedAttrList writeEn, clk, reset, done;
  writeEn.append("go", isSet);
  clk.append("clk", isSet);
  done.append("done", isSet);
  portAttributes.append({DictionaryAttr(),               // In
                         writeEn.getDictionary(context), // Write enable
                         clk.getDictionary(context),     // Clk
                         DictionaryAttr(),               // Out
                         done.getDictionary(context)}    // Done
  );
  return portAttributes;
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
    types.push_back(builder.getIntegerType(size)); // Addresses
  types.push_back(builder.getIntegerType(width));  // Write data
  types.push_back(builder.getI1Type());            // Write enable
  types.push_back(builder.getI1Type());            // Clk
  types.push_back(builder.getIntegerType(width));  // Read data
  types.push_back(builder.getI1Type());            // Done
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

  size_t numExtraPorts = 5; // write data/enable, clk, and read data/done.
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
  StringRef groupName = enableOp.groupName();

  auto groupOp = wiresOp.lookupSymbol<GroupInterface>(groupName);
  if (!groupOp)
    return enableOp.emitOpError()
           << "with group '" << groupName << "', which does not exist.";

  if (isa<CombGroupOp>(groupOp))
    return enableOp.emitOpError() << "with group '" << groupName
                                  << "', which is a combinational group.";

  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyIfOp(IfOp ifOp) {
  auto component = ifOp->getParentOfType<ComponentOp>();
  auto wiresOp = component.getWiresOp();

  if (ifOp.thenRegion().front().empty())
    return ifOp.emitError() << "empty 'then' region.";

  if (ifOp.elseRegion().getBlocks().size() != 0 &&
      ifOp.elseRegion().front().empty())
    return ifOp.emitError() << "empty 'else' region.";

  Optional<StringRef> optGroupName = ifOp.groupName();
  if (!optGroupName.hasValue()) {
    /// No combinational group was provided
    return success();
  }
  StringRef groupName = optGroupName.getValue();
  auto groupOp = wiresOp.lookupSymbol<GroupInterface>(groupName);
  if (!groupOp)
    return ifOp.emitOpError()
           << "with group '" << groupName << "', which does not exist.";

  if (isa<GroupOp>(groupOp))
    return ifOp.emitOpError() << "with group '" << groupName
                              << "', which is not a combinational group.";

  if (failed(portDrivenByGroup(ifOp.cond(), groupOp)))
    return ifOp.emitError()
           << "with conditional op: '" << valueName(component, ifOp.cond())
           << "' expected to be driven from group: '" << groupName
           << "' but no driver was found.";

  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyWhileOp(WhileOp whileOp) {
  auto component = whileOp->getParentOfType<ComponentOp>();
  auto wiresOp = component.getWiresOp();

  if (whileOp.body().front().empty())
    return whileOp.emitError() << "empty body region.";

  Optional<StringRef> optGroupName = whileOp.groupName();
  if (!optGroupName.hasValue()) {
    /// No combinational group was provided
    return success();
  }
  StringRef groupName = optGroupName.getValue();
  auto groupOp = wiresOp.lookupSymbol<GroupInterface>(groupName);
  if (!groupOp)
    return whileOp.emitOpError()
           << "with group '" << groupName << "', which does not exist.";

  if (isa<GroupOp>(groupOp))
    return whileOp.emitOpError() << "with group '" << groupName
                                 << "', which is not a combinational group.";

  if (failed(portDrivenByGroup(whileOp.cond(), groupOp)))
    return whileOp.emitError()
           << "conditional op: '" << valueName(component, whileOp.cond())
           << "' expected to be driven from group: '" << groupName
           << "' but no driver was found.";

  return success();
}

//===----------------------------------------------------------------------===//
// Calyx library ops
//===----------------------------------------------------------------------===//

#define ImplUnaryOpCellInterface(OpType)                                       \
  SmallVector<StringRef> OpType::portNames() { return {"in", "out"}; }         \
  SmallVector<Direction> OpType::portDirections() { return {Input, Output}; }  \
  SmallVector<DictionaryAttr> OpType::portAttributes() {                       \
    return {DictionaryAttr(), DictionaryAttr()};                               \
  }                                                                            \
  void OpType::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {              \
    getCellAsmResultNames(setNameFn, *this, this->portNames());                \
  }

#define ImplBinOpCellInterface(OpType)                                         \
  SmallVector<StringRef> OpType::portNames() {                                 \
    return {"left", "right", "out"};                                           \
  }                                                                            \
  SmallVector<Direction> OpType::portDirections() {                            \
    return {Input, Input, Output};                                             \
  }                                                                            \
  void OpType::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {              \
    getCellAsmResultNames(setNameFn, *this, this->portNames());                \
  }                                                                            \
  SmallVector<DictionaryAttr> OpType::portAttributes() {                       \
    return {DictionaryAttr(), DictionaryAttr(), DictionaryAttr()};             \
  }

// clang-format off
ImplUnaryOpCellInterface(PadLibOp)
ImplUnaryOpCellInterface(SliceLibOp)
ImplUnaryOpCellInterface(NotLibOp)

ImplBinOpCellInterface(LtLibOp)
ImplBinOpCellInterface(GtLibOp)
ImplBinOpCellInterface(EqLibOp)
ImplBinOpCellInterface(NeqLibOp)
ImplBinOpCellInterface(GeLibOp)
ImplBinOpCellInterface(LeLibOp)
ImplBinOpCellInterface(SltLibOp)
ImplBinOpCellInterface(SgtLibOp)
ImplBinOpCellInterface(SeqLibOp)
ImplBinOpCellInterface(SneqLibOp)
ImplBinOpCellInterface(SgeLibOp)
ImplBinOpCellInterface(SleLibOp)

ImplBinOpCellInterface(AddLibOp)
ImplBinOpCellInterface(SubLibOp)
ImplBinOpCellInterface(ShruLibOp)
ImplBinOpCellInterface(RshLibOp)
ImplBinOpCellInterface(SrshLibOp)
ImplBinOpCellInterface(LshLibOp)
ImplBinOpCellInterface(AndLibOp)
ImplBinOpCellInterface(OrLibOp)
ImplBinOpCellInterface(XorLibOp)
// clang-format on

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxInterfaces.cpp.inc"

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Calyx/Calyx.cpp.inc"
