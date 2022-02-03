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

Direction direction::get(bool isOutput) {
  return static_cast<Direction>(isOutput);
}

IntegerAttr direction::packAttribute(MLIRContext *ctx, size_t nIns,
                                     size_t nOuts) {
  // Pack the array of directions into an APInt.  Input direction is zero,
  // output direction is one.
  size_t numDirections = nIns + nOuts;
  APInt portDirections(/*width=*/numDirections, /*value=*/0);
  for (size_t i = nIns, e = numDirections; i != e; ++i)
    portDirections.setBit(i);

  return IntegerAttr::get(IntegerType::get(ctx, numDirections), portDirections);
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// This pattern collapses a calyx.seq or calyx.par operation when it
/// contains exactly one calyx.enable operation.
template <typename CtrlOp>
struct CollapseUnaryControl : mlir::OpRewritePattern<CtrlOp> {
  using mlir::OpRewritePattern<CtrlOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp ctrlOp,
                                PatternRewriter &rewriter) const override {
    auto &ops = ctrlOp.getBody()->getOperations();
    bool isUnaryControl = (ops.size() == 1) && isa<EnableOp>(ops.front()) &&
                          isa<SeqOp, ParOp>(ctrlOp->getParentOp());
    if (!isUnaryControl)
      return failure();

    ops.front().moveBefore(ctrlOp);
    rewriter.eraseOp(ctrlOp);
    return success();
  }
};

/// Verify that the value is not a "complex" value. For example, the source
/// of an AssignOp should be a constant or port, e.g.
/// %and = comb.and %a, %b : i1
/// calyx.assign %port = %c1_i1 ? %and   : i1   // Incorrect
/// calyx.assign %port = %and   ? %c1_i1 : i1   // Correct
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

/// Convenience function for getting the SSA name of `v` under the scope of
/// operation `scopeOp`.
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

// Helper function for parsing a group port operation, i.e. GroupDoneOp and
// GroupPortOp. These may take one of two different forms:
// (1) %<guard> ? %<src> : i1
// (2) %<src> : i1
static ParseResult parseGroupPort(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> operandInfos;
  OpAsmParser::OperandType guardOrSource;
  if (parser.parseOperand(guardOrSource))
    return failure();

  if (succeeded(parser.parseOptionalQuestion())) {
    OpAsmParser::OperandType source;
    // The guard exists.
    if (parser.parseOperand(source))
      return failure();
    operandInfos.push_back(source);
  }
  // No matter if this is the source or guard, it should be last.
  operandInfos.push_back(guardOrSource);

  Type type;
  // Resolving the operands with the same type works here since the source and
  // guard of a group port is always i1.
  if (parser.parseColonType(type) ||
      parser.resolveOperands(operandInfos, type, result.operands))
    return failure();

  return success();
}

// A helper function for printing group ports, i.e. GroupGoOp and GroupDoneOp.
template <typename GroupPortType>
static void printGroupPort(OpAsmPrinter &p, GroupPortType op) {
  static_assert(std::is_same<GroupGoOp, GroupPortType>() ||
                    std::is_same<GroupDoneOp, GroupPortType>(),
                "Should be a Calyx Group port.");

  p << " ";
  // The guard is optional.
  Value guard = op.guard(), source = op.src();
  if (guard)
    p << guard << " ? ";
  p << source << " : " << source.getType();
}

// Collapse nested control of the same type for SeqOp and ParOp, e.g.
// calyx.seq { calyx.seq { ... } } -> calyx.seq { ... }
template <typename OpTy>
static LogicalResult collapseControl(OpTy controlOp,
                                     PatternRewriter &rewriter) {
  static_assert(std::is_same<SeqOp, OpTy>() || std::is_same<ParOp, OpTy>(),
                "Should be a SeqOp or ParOp.");

  if (isa<OpTy>(controlOp->getParentOp())) {
    Block *controlBody = controlOp.getBody();
    for (auto &op : make_early_inc_range(*controlBody))
      op.moveBefore(controlOp);

    rewriter.eraseOp(controlOp);
    return success();
  }

  return failure();
}

template <typename OpTy>
static LogicalResult emptyControl(OpTy controlOp, PatternRewriter &rewriter) {
  if (controlOp.getBody()->empty()) {
    rewriter.eraseOp(controlOp);
    return success();
  }
  return failure();
}

/// A helper function to check whether the conditional and group (if it exists)
/// needs to be erased to maintain a valid state of a Calyx program. If these
/// have no more uses, they will be erased.
template <typename OpTy>
static void eraseControlWithGroupAndConditional(OpTy op,
                                                PatternRewriter &rewriter) {
  static_assert(std::is_same<OpTy, IfOp>() || std::is_same<OpTy, WhileOp>(),
                "This is only applicable to WhileOp and IfOp.");

  // Save information about the operation, and erase it.
  Value cond = op.cond();
  Optional<StringRef> groupName = op.groupName();
  auto component = op->template getParentOfType<ComponentOp>();
  rewriter.eraseOp(op);

  // Clean up the attached conditional and combinational group (if it exists).
  if (groupName.hasValue()) {
    auto group = component.getWiresOp().template lookupSymbol<GroupInterface>(
        *groupName);
    if (SymbolTable::symbolKnownUseEmpty(group, component.getRegion()))
      rewriter.eraseOp(group);
  }
  // Check the conditional after the Group, since it will be driven within.
  if (!cond.isa<BlockArgument>() && cond.getDefiningOp()->use_empty())
    rewriter.eraseOp(cond.getDefiningOp());
}

//===----------------------------------------------------------------------===//
// ProgramOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyProgramOp(ProgramOp program) {
  if (program.getEntryPointComponent() == nullptr)
    return program.emitOpError() << "has undefined entry-point component: \""
                                 << program.entryPointName() << "\".";

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
  ArrayAttr portNamesAttr = portNames(), portAttrs = portAttributes();
  APInt portDirectionsAttr = portDirections();

  SmallVector<PortInfo> results;
  for (size_t i = 0, e = portNamesAttr.size(); i != e; ++i) {
    results.push_back(PortInfo{portNamesAttr[i].cast<StringAttr>(),
                               portTypes[i],
                               direction::get(portDirectionsAttr[i]),
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

  p << " ";
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false,
                /*printEmptyBlock=*/false);

  SmallVector<StringRef> elidedAttrs = {"portAttributes", "portNames",
                                        "portDirections", "sym_name", "type"};
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

/// Parses the ports of a Calyx component signature, and adds the corresponding
/// port names to `attrName`.
static ParseResult
parsePortDefList(OpAsmParser &parser, OperationState &result,
                 SmallVectorImpl<OpAsmParser::OperandType> &ports,
                 SmallVectorImpl<Type> &portTypes,
                 SmallVectorImpl<NamedAttrList> &portAttrs) {
  auto parsePort = [&]() -> ParseResult {
    OpAsmParser::OperandType port;
    Type portType;
    // Expect each port to have the form `%<ssa-name> : <type>`.
    if (parser.parseRegionArgument(port) || parser.parseColon() ||
        parser.parseType(portType))
      return failure();
    ports.push_back(port);
    portTypes.push_back(portType);

    NamedAttrList portAttr;
    portAttrs.push_back(succeeded(parser.parseOptionalAttrDict(portAttr))
                            ? portAttr
                            : NamedAttrList());
    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parsePort);
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
      "portDirections",
      direction::packAttribute(context, inPorts.size(), outPorts.size()));

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

/// Returns a new vector containing the concatenation of vectors `a` and `b`.
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
  result.addAttribute("portDirections",
                      direction::packAttribute(builder.getContext(),
                                               portIOTypes.first.size(),
                                               portIOTypes.second.size()));
  // Record the attributes of the ports.
  result.addAttribute("portAttributes", builder.getArrayAttr(portAttributes));

  // Create a single-blocked region.
  Region *region = result.addRegion();
  Block *body = new Block();
  region->push_back(body);

  // Add all ports to the body.
  body->addArguments(portTypes);

  // Insert the WiresOp and ControlOp.
  IRRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  builder.create<WiresOp>(result.location);
  builder.create<ControlOp>(result.location);
}

void ComponentOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto ports = portNames();
  auto *block = &getRegion()->front();
  for (size_t i = 0, e = block->getNumArguments(); i != e; ++i)
    setNameFn(block->getArgument(i), ports[i].cast<StringAttr>().getValue());
}

//===----------------------------------------------------------------------===//
// ControlOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyControlOp(ControlOp control) {
  return verifyControlBody(control);
}

//===----------------------------------------------------------------------===//
// SeqOp
//===----------------------------------------------------------------------===//

void SeqOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add(collapseControl<SeqOp>);
  patterns.add(emptyControl<SeqOp>);
  patterns.insert<CollapseUnaryControl<SeqOp>>(context);
}

//===----------------------------------------------------------------------===//
// ParOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyParOp(ParOp parOp) {
  llvm::SmallSet<StringRef, 8> groupNames;
  Block *body = parOp.getBody();

  // Add loose requirement that the body of a ParOp may not enable the same
  // Group more than once, e.g. calyx.par { calyx.enable @G calyx.enable @G }
  for (EnableOp op : body->getOps<EnableOp>()) {
    StringRef groupName = op.groupName();
    if (groupNames.count(groupName))
      return parOp->emitOpError() << "cannot enable the same group: \""
                                  << groupName << "\" more than once.";
    groupNames.insert(groupName);
  }

  return success();
}

void ParOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add(collapseControl<ParOp>);
  patterns.add(emptyControl<ParOp>);
  patterns.insert<CollapseUnaryControl<ParOp>>(context);
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
    return value == r.out()
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
// GroupInterface
//===----------------------------------------------------------------------===//

/// Determines whether the given port is used in the group. Its use depends on
/// the `isDriven` value; if true, then the port should be a destination in an
/// AssignOp. Otherwise, it should be the source, i.e. a read.
static bool portIsUsedInGroup(GroupInterface group, Value port, bool isDriven) {
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

/// Checks whether `port` is driven from within `groupOp`.
static LogicalResult portDrivenByGroup(GroupInterface groupOp, Value port) {
  // Check if the port is driven by an assignOp from within `groupOp`.
  if (portIsUsedInGroup(groupOp, port, /*isDriven=*/true))
    return success();

  // If `port` is an output of a cell then we conservatively enforce that at
  // least one input port of the cell must be driven by the group.
  if (auto cell = dyn_cast<CellInterface>(port.getDefiningOp());
      cell && cell.direction(port) == calyx::Direction::Output)
    return groupOp.drivesAnyPort(cell.getInputPorts());

  return failure();
}

LogicalResult GroupOp::drivesPort(Value port) {
  return portDrivenByGroup(*this, port);
}

LogicalResult CombGroupOp::drivesPort(Value port) {
  return portDrivenByGroup(*this, port);
}

/// Checks whether all ports are driven within the group.
static LogicalResult allPortsDrivenByGroup(GroupInterface group,
                                           ValueRange ports) {
  return success(llvm::all_of(ports, [&](Value port) {
    return portIsUsedInGroup(group, port, /*isDriven=*/true);
  }));
}

LogicalResult GroupOp::drivesAllPorts(ValueRange ports) {
  return allPortsDrivenByGroup(*this, ports);
}

LogicalResult CombGroupOp::drivesAllPorts(ValueRange ports) {
  return allPortsDrivenByGroup(*this, ports);
}

/// Checks whether any ports are driven within the group.
static LogicalResult anyPortsDrivenByGroup(GroupInterface group,
                                           ValueRange ports) {
  return success(llvm::any_of(ports, [&](Value port) {
    return portIsUsedInGroup(group, port, /*isDriven=*/true);
  }));
}

LogicalResult GroupOp::drivesAnyPort(ValueRange ports) {
  return anyPortsDrivenByGroup(*this, ports);
}

LogicalResult CombGroupOp::drivesAnyPort(ValueRange ports) {
  return anyPortsDrivenByGroup(*this, ports);
}

/// Checks whether any ports are read within the group.
static LogicalResult anyPortsReadByGroup(GroupInterface group,
                                         ValueRange ports) {
  return success(llvm::any_of(ports, [&](Value port) {
    return portIsUsedInGroup(group, port, /*isDriven=*/false);
  }));
}

LogicalResult GroupOp::readsAnyPort(ValueRange ports) {
  return anyPortsReadByGroup(*this, ports);
}

LogicalResult CombGroupOp::readsAnyPort(ValueRange ports) {
  return anyPortsReadByGroup(*this, ports);
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
            return succeeded(group.drivesAnyPort({op.write_en(), op.in()}))
                       ? group.drivesAllPorts({op.write_en(), op.in()})
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
            return succeeded(
                       group.drivesAnyPort({op.writeData(), op.writeEn()}))
                       ? group.drivesAllPorts(requiredWritePorts)
                       : success();
          })
          .Case<AndLibOp, OrLibOp, XorLibOp, AddLibOp, SubLibOp, GtLibOp,
                LtLibOp, EqLibOp, NeqLibOp, GeLibOp, LeLibOp, LshLibOp,
                RshLibOp, SgtLibOp, SltLibOp, SeqLibOp, SneqLibOp, SgeLibOp,
                SleLibOp, SrshLibOp>([&](auto op) {
            Value lhs = op.left(), rhs = op.right();
            return succeeded(group.drivesAnyPort({lhs, rhs}))
                       ? group.drivesAllPorts({lhs, rhs})
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
            return succeeded(group.readsAnyPort({op.readData()}))
                       ? group.drivesAllPorts(op.addrPorts())
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

//===----------------------------------------------------------------------===//
// Utilities for operations with the Cell trait.
//===----------------------------------------------------------------------===//

/// Gives each result of the cell a meaningful name in the form:
/// <instance-name>.<port-name>
static void getCellAsmResultNames(OpAsmSetValueNameFn setNameFn, Operation *op,
                                  ArrayRef<StringRef> portNames) {
  assert(isa<CellInterface>(op) && "must implement the Cell interface");

  auto instanceName =
      op->getAttrOfType<FlatSymbolRefAttr>("instanceName").getValue();
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

static ParseResult parseAssignOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType destination;
  if (parser.parseOperand(destination) || parser.parseEqual())
    return failure();

  // An AssignOp takes one of the two following forms:
  // (1) %<dest> = %<src> : <type>
  // (2) %<dest> = %<guard> ? %<src> : <type>
  OpAsmParser::OperandType guardOrSource;
  if (parser.parseOperand(guardOrSource))
    return failure();

  // Since the guard is optional, we need to check if there is an accompanying
  // `?` symbol.
  OpAsmParser::OperandType source;
  bool hasGuard = succeeded(parser.parseOptionalQuestion());
  if (hasGuard) {
    // The guard exists. Parse the source.
    if (parser.parseOperand(source))
      return failure();
  }

  Type type;
  if (parser.parseColonType(type) ||
      parser.resolveOperand(destination, type, result.operands))
    return failure();

  if (hasGuard) {
    Type i1Type = parser.getBuilder().getI1Type();
    // Since the guard is optional, it is listed last in the arguments of the
    // AssignOp. Therefore, we must parse the source first.
    if (parser.resolveOperand(source, type, result.operands) ||
        parser.resolveOperand(guardOrSource, i1Type, result.operands))
      return failure();
  } else {
    // This is actually a source.
    if (parser.resolveOperand(guardOrSource, type, result.operands))
      return failure();
  }

  return success();
}

static void printAssignOp(OpAsmPrinter &p, AssignOp op) {
  p << " " << op.dest() << " = ";

  Value guard = op.guard(), source = op.src();
  // The guard is optional.
  if (guard)
    p << guard << " ? ";

  // We only need to print a single type; the destination and source are
  // guaranteed to be the same type.
  p << source << " : " << source.getType();
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
  auto program = instance->getParentOfType<ProgramOp>();
  StringRef entryPointName = program.entryPointName();
  if (instance.componentName() == entryPointName)
    return instance.emitOpError()
           << "cannot reference the entry-point component: '" << entryPointName
           << "'.";

  // Verify there are no other instances with this name.
  auto component = instance->getParentOfType<ComponentOp>();
  StringAttr name =
      StringAttr::get(instance.getContext(), instance.instanceName());
  Optional<SymbolTable::UseRange> componentUseRange =
      SymbolTable::getSymbolUses(name, component.getRegion());
  if (componentUseRange.hasValue() &&
      llvm::any_of(componentUseRange.getValue(),
                   [&](SymbolTable::SymbolUse use) {
                     return use.getUser() != instance;
                   }))
    return instance.emitOpError()
           << "with instance symbol: '" << name.getValue()
           << "' is already a symbol for another instance.";

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
  Operation *op = *this;
  auto program = op->getParentOfType<ProgramOp>();
  Operation *referencedComponent =
      symbolTable.lookupNearestSymbolFrom(program, componentNameAttr());
  if (referencedComponent == nullptr)
    return emitError() << "referencing component: '" << componentName()
                       << "', which does not exist.";

  Operation *shadowedComponentName =
      symbolTable.lookupNearestSymbolFrom(program, instanceNameAttr());
  if (shadowedComponentName != nullptr)
    return emitError() << "instance symbol: '" << instanceName()
                       << "' is already a symbol for another component.";

  // Verify the referenced component is not instantiating itself.
  auto parentComponent = op->getParentOfType<ComponentOp>();
  if (parentComponent == referencedComponent)
    return emitError() << "recursive instantiation of its parent component: '"
                       << componentName() << "'";

  assert(isa<ComponentOp>(referencedComponent) && "Should be a ComponentOp.");
  return verifyInstanceOpType(*this, cast<ComponentOp>(referencedComponent));
}

/// Provide meaningful names to the result values of an InstanceOp.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

SmallVector<StringRef> InstanceOp::portNames() {
  SmallVector<StringRef> portNames;
  for (Attribute name : getReferencedComponent().portNames())
    portNames.push_back(name.cast<StringAttr>().getValue());
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
  StringRef name = parent.sym_name();
  std::string resultName = name.str() + ".go";
  setNameFn(getResult(), resultName);
}

static void printGroupGoOp(OpAsmPrinter &p, GroupGoOp op) {
  printGroupPort(p, op);
}

static ParseResult parseGroupGoOp(OpAsmParser &parser, OperationState &result) {
  if (parseGroupPort(parser, result))
    return failure();

  result.addTypes(parser.getBuilder().getI1Type());
  return success();
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

static void printGroupDoneOp(OpAsmPrinter &p, GroupDoneOp op) {
  printGroupPort(p, op);
}

static ParseResult parseGroupDoneOp(OpAsmParser &parser,
                                    OperationState &result) {
  return parseGroupPort(parser, result);
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
                     StringRef instanceName, int64_t width,
                     ArrayRef<int64_t> sizes, ArrayRef<int64_t> addrSizes) {
  state.addAttribute("instanceName", FlatSymbolRefAttr::get(
                                         builder.getContext(), instanceName));
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
  ArrayRef<Attribute> sizes = memoryOp.sizes().getValue();
  ArrayRef<Attribute> addrSizes = memoryOp.addrSizes().getValue();
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
  WiresOp wiresOp = component.getWiresOp();

  if (ifOp.elseBodyExists() && ifOp.getElseBody()->empty())
    return ifOp.emitError() << "empty 'else' region.";

  Optional<StringRef> optGroupName = ifOp.groupName();
  if (!optGroupName.hasValue()) {
    // No combinational group was provided.
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

  if (failed(groupOp.drivesPort(ifOp.cond())))
    return ifOp.emitError()
           << "with conditional op: '" << valueName(component, ifOp.cond())
           << "' expected to be driven from group: '" << groupName
           << "' but no driver was found.";

  return success();
}

/// Returns the last EnableOp within the child tree of 'parentSeqOp'. If no
/// EnableOp was found (e.g. a "calyx.par" operation is present), returns
/// None.
static Optional<EnableOp> getLastEnableOp(SeqOp parent) {
  auto &lastOp = parent.getBody()->back();
  if (auto enableOp = dyn_cast<EnableOp>(lastOp))
    return enableOp;
  else if (auto seqOp = dyn_cast<SeqOp>(lastOp))
    return getLastEnableOp(seqOp);

  return None;
}

/// Returns a mapping of {enabled Group name, EnableOp} for all EnableOps within
/// the immediate ParOp's body.
static llvm::StringMap<EnableOp> getAllEnableOpsInImmediateBody(ParOp parent) {
  llvm::StringMap<EnableOp> enables;
  Block *body = parent.getBody();
  for (EnableOp op : body->getOps<EnableOp>())
    enables.insert(std::pair(op.groupName(), op));

  return enables;
}

/// Checks preconditions for the common tail pattern. This canonicalization is
/// stringent about not entering nested control operations, as this may cause
/// unintentional changes in behavior.
/// We only look for two cases: (1) both regions are ParOps, and
/// (2) both regions are SeqOps. The case when these are different, e.g. ParOp
/// and SeqOp, will only produce less optimal code, or even worse, change the
/// behavior.
template <typename OpTy>
static bool hasCommonTailPatternPreConditions(IfOp op) {
  static_assert(std::is_same<SeqOp, OpTy>() || std::is_same<ParOp, OpTy>(),
                "Should be a SeqOp or ParOp.");

  if (!op.thenBodyExists() || !op.elseBodyExists())
    return false;
  if (op.getThenBody()->empty() || op.getElseBody()->empty())
    return false;

  Block *thenBody = op.getThenBody(), *elseBody = op.getElseBody();
  return isa<OpTy>(thenBody->front()) && isa<OpTy>(elseBody->front());
}

///                                         seq {
///   if %a with @G {                         if %a with @G {
///     seq { ... calyx.enable @A }             seq { ... }
///   else {                          ->      } else {
///     seq { ... calyx.enable @A }             seq { ... }
///   }                                       }
///                                           calyx.enable @A
///                                         }
struct CommonTailPatternWithSeq : mlir::OpRewritePattern<IfOp> {
  using mlir::OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!hasCommonTailPatternPreConditions<SeqOp>(ifOp))
      return failure();

    auto thenControl = cast<SeqOp>(ifOp.getThenBody()->front()),
         elseControl = cast<SeqOp>(ifOp.getElseBody()->front());
    Optional<EnableOp> lastThenEnableOp = getLastEnableOp(thenControl),
                       lastElseEnableOp = getLastEnableOp(elseControl);

    if (!lastThenEnableOp.hasValue() || !lastElseEnableOp.hasValue())
      return failure();
    if (lastThenEnableOp->groupName() != lastElseEnableOp->groupName())
      return failure();

    // Place the IfOp and pulled EnableOp inside a sequential region, in case
    // this IfOp is nested in a ParOp. This avoids unintentionally
    // parallelizing the pulled out EnableOps.
    rewriter.setInsertionPointAfter(ifOp);
    SeqOp seqOp = rewriter.create<SeqOp>(ifOp.getLoc());
    Block *body = seqOp.getBody();
    ifOp->remove();
    body->push_back(ifOp);
    rewriter.setInsertionPointToEnd(body);
    rewriter.create<EnableOp>(seqOp.getLoc(), lastThenEnableOp->groupName());

    // Erase the common EnableOp from the Then and Else regions.
    rewriter.eraseOp(*lastThenEnableOp);
    rewriter.eraseOp(*lastElseEnableOp);
    return success();
  }
};

///    if %a with @G {              par {
///      par {                        if %a with @G {
///        ...                          par { ... }
///        calyx.enable @A            } else {
///        calyx.enable @B    ->        par { ... }
///      }                            }
///    } else {                       calyx.enable @A
///      par {                        calyx.enable @B
///        ...                      }
///        calyx.enable @A
///        calyx.enable @B
///      }
///    }
struct CommonTailPatternWithPar : mlir::OpRewritePattern<IfOp> {
  using mlir::OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!hasCommonTailPatternPreConditions<ParOp>(ifOp))
      return failure();
    auto thenControl = cast<ParOp>(ifOp.getThenBody()->front()),
         elseControl = cast<ParOp>(ifOp.getElseBody()->front());

    llvm::StringMap<EnableOp> A = getAllEnableOpsInImmediateBody(thenControl),
                              B = getAllEnableOpsInImmediateBody(elseControl);

    // Compute the intersection between `A` and `B`.
    SmallVector<StringRef> groupNames;
    for (auto a = A.begin(); a != A.end(); ++a) {
      StringRef groupName = a->getKey();
      auto b = B.find(groupName);
      if (b == B.end())
        continue;
      // This is also an element in B.
      groupNames.push_back(groupName);
      // Since these are being pulled out, erase them.
      rewriter.eraseOp(a->getValue());
      rewriter.eraseOp(b->getValue());
    }
    // Place the IfOp and EnableOp(s) inside a parallel region, in case this
    // IfOp is nested in a SeqOp. This avoids unintentionally sequentializing
    // the pulled out EnableOps.
    rewriter.setInsertionPointAfter(ifOp);
    ParOp parOp = rewriter.create<ParOp>(ifOp.getLoc());
    Block *body = parOp.getBody();
    ifOp->remove();
    body->push_back(ifOp);

    // Pull out the intersection between these two sets, and erase their
    // counterparts in the Then and Else regions.
    rewriter.setInsertionPointToEnd(body);
    for (StringRef groupName : groupNames)
      rewriter.create<EnableOp>(parOp.getLoc(), groupName);

    return success();
  }
};

/// This pattern checks for one of two cases that will lead to IfOp deletion:
/// (1) Then and Else bodies are both empty.
/// (2) Then body is empty and Else body does not exist.
struct EmptyIfBody : mlir::OpRewritePattern<IfOp> {
  using mlir::OpRewritePattern<IfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!ifOp.getThenBody()->empty())
      return failure();
    if (ifOp.elseBodyExists() && !ifOp.getElseBody()->empty())
      return failure();

    eraseControlWithGroupAndConditional(ifOp, rewriter);

    return success();
  }
};

void IfOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<CommonTailPatternWithSeq, CommonTailPatternWithPar, EmptyIfBody>(
      context);
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyWhileOp(WhileOp whileOp) {
  auto component = whileOp->getParentOfType<ComponentOp>();
  auto wiresOp = component.getWiresOp();

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

  if (failed(groupOp.drivesPort(whileOp.cond())))
    return whileOp.emitError()
           << "conditional op: '" << valueName(component, whileOp.cond())
           << "' expected to be driven from group: '" << groupName
           << "' but no driver was found.";

  return success();
}

LogicalResult WhileOp::canonicalize(WhileOp whileOp,
                                    PatternRewriter &rewriter) {
  if (whileOp.getBody()->empty()) {
    eraseControlWithGroupAndConditional(whileOp, rewriter);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// MultPipe
//===----------------------------------------------------------------------===//

SmallVector<StringRef> MultPipeLibOp::portNames() {
  return {"left", "right", "go", "clk", "reset", "out", "done"};
}

SmallVector<Direction> MultPipeLibOp::portDirections() {
  return {Input, Input, Input, Input, Input, Output, Output};
}

void MultPipeLibOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

SmallVector<DictionaryAttr> MultPipeLibOp::portAttributes() {
  MLIRContext *context = getContext();
  IntegerAttr isSet = IntegerAttr::get(IntegerType::get(context, 1), 1);
  NamedAttrList go, clk, reset, done;
  go.append("go", isSet);
  clk.append("clk", isSet);
  reset.append("reset", isSet);
  done.append("done", isSet);
  return {
      DictionaryAttr(),             /* Lhs    */
      DictionaryAttr(),             /* Rhs    */
      go.getDictionary(context),    /* Go     */
      clk.getDictionary(context),   /* Clk    */
      reset.getDictionary(context), /* Reset  */
      DictionaryAttr(),             /* Out    */
      done.getDictionary(context)   /* Done   */
  };
}

//===----------------------------------------------------------------------===//
// DivPipe
//===----------------------------------------------------------------------===//

SmallVector<StringRef> DivPipeLibOp::portNames() {
  return {"left",         "right",         "go",  "clk", "reset",
          "out_quotient", "out_remainder", "done"};
}

SmallVector<Direction> DivPipeLibOp::portDirections() {
  return {Input, Input, Input, Input, Input, Output, Output, Output};
}

void DivPipeLibOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

SmallVector<DictionaryAttr> DivPipeLibOp::portAttributes() {
  MLIRContext *context = getContext();
  IntegerAttr isSet = IntegerAttr::get(IntegerType::get(context, 1), 1);
  NamedAttrList go, clk, reset, done;
  go.append("go", isSet);
  clk.append("clk", isSet);
  reset.append("reset", isSet);
  done.append("done", isSet);
  return {
      DictionaryAttr(),             /* Lhs       */
      DictionaryAttr(),             /* Rhs       */
      go.getDictionary(context),    /* Go        */
      clk.getDictionary(context),   /* Clk       */
      reset.getDictionary(context), /* Reset     */
      DictionaryAttr(),             /* Quotient  */
      DictionaryAttr(),             /* Remainder */
      done.getDictionary(context)   /* Done      */
  };
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
