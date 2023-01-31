//===- FIRRTLOps.cpp - Implement the FIRRTL operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the FIRRTL ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using llvm::SmallDenseSet;
using mlir::RegionRange;
using namespace circt;
using namespace firrtl;
using namespace chirrtl;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Remove elements from the input array corresponding to set bits in
/// `indicesToDrop`, returning the elements not mentioned.
template <typename T>
static SmallVector<T>
removeElementsAtIndices(ArrayRef<T> input,
                        const llvm::BitVector &indicesToDrop) {
#ifndef NDEBUG
  if (!input.empty()) {
    int lastIndex = indicesToDrop.find_last();
    if (lastIndex >= 0)
      assert((size_t)lastIndex < input.size() && "index out of range");
  }
#endif

  // If the input is empty (which is an optimization we do for certain array
  // attributes), simply return an empty vector.
  if (input.empty())
    return {};

  // Copy over the live chunks.
  size_t lastCopied = 0;
  SmallVector<T> result;
  result.reserve(input.size() - indicesToDrop.count());

  for (unsigned indexToDrop : indicesToDrop.set_bits()) {
    // If we skipped over some valid elements, copy them over.
    if (indexToDrop > lastCopied) {
      result.append(input.begin() + lastCopied, input.begin() + indexToDrop);
      lastCopied = indexToDrop;
    }
    // Ignore this value so we don't copy it in the next iteration.
    ++lastCopied;
  }

  // If there are live elements at the end, copy them over.
  if (lastCopied < input.size())
    result.append(input.begin() + lastCopied, input.end());

  return result;
}

bool firrtl::isDuplexValue(Value val) {
  Operation *op = val.getDefiningOp();
  // Block arguments are not duplex values.
  if (!op)
    return false;
  return TypeSwitch<Operation *, bool>(op)
      .Case<SubfieldOp, SubindexOp, SubaccessOp>(
          [](auto op) { return isDuplexValue(op.getInput()); })
      .Case<RegOp, RegResetOp, WireOp>([](auto) { return true; })
      .Default([](auto) { return false; });
}

/// Return the kind of port this is given the port type from a 'mem' decl.
static MemOp::PortKind getMemPortKindFromType(FIRRTLType type) {
  constexpr unsigned int addr = 1 << 0;
  constexpr unsigned int en = 1 << 1;
  constexpr unsigned int clk = 1 << 2;
  constexpr unsigned int data = 1 << 3;
  constexpr unsigned int mask = 1 << 4;
  constexpr unsigned int rdata = 1 << 5;
  constexpr unsigned int wdata = 1 << 6;
  constexpr unsigned int wmask = 1 << 7;
  constexpr unsigned int wmode = 1 << 8;
  constexpr unsigned int def = 1 << 9;
  // Get the kind of port based on the fields of the Bundle.
  auto portType = type.dyn_cast<BundleType>();
  if (!portType)
    return MemOp::PortKind::Debug;
  unsigned fields = 0;
  // Get the kind of port based on the fields of the Bundle.
  for (auto elem : portType.getElements()) {
    fields |= llvm::StringSwitch<unsigned>(elem.name.getValue())
                  .Case("addr", addr)
                  .Case("en", en)
                  .Case("clk", clk)
                  .Case("data", data)
                  .Case("mask", mask)
                  .Case("rdata", rdata)
                  .Case("wdata", wdata)
                  .Case("wmask", wmask)
                  .Case("wmode", wmode)
                  .Default(def);
  }
  if (fields == (addr | en | clk | data))
    return MemOp::PortKind::Read;
  if (fields == (addr | en | clk | data | mask))
    return MemOp::PortKind::Write;
  if (fields == (addr | en | clk | wdata | wmask | rdata | wmode))
    return MemOp::PortKind::ReadWrite;
  return MemOp::PortKind::Debug;
}

Flow firrtl::swapFlow(Flow flow) {
  switch (flow) {
  case Flow::Source:
    return Flow::Sink;
  case Flow::Sink:
    return Flow::Source;
  case Flow::Duplex:
    return Flow::Duplex;
  }
  llvm_unreachable("invalid flow");
}

Flow firrtl::foldFlow(Value val, Flow accumulatedFlow) {
  auto swap = [&accumulatedFlow]() -> Flow {
    return swapFlow(accumulatedFlow);
  };

  if (auto blockArg = val.dyn_cast<BlockArgument>()) {
    auto op = val.getParentBlock()->getParentOp();
    auto direction =
        cast<FModuleLike>(op).getPortDirection(blockArg.getArgNumber());
    if (direction == Direction::Out)
      return swap();
    return accumulatedFlow;
  }

  Operation *op = val.getDefiningOp();

  return TypeSwitch<Operation *, Flow>(op)
      .Case<SubfieldOp>([&](auto op) {
        return foldFlow(op.getInput(),
                        op.isFieldFlipped() ? swap() : accumulatedFlow);
      })
      .Case<SubindexOp, SubaccessOp, RefSubOp>(
          [&](auto op) { return foldFlow(op.getInput(), accumulatedFlow); })
      // Registers, Wires, and behavioral memory ports are always Duplex.
      .Case<RegOp, RegResetOp, WireOp, MemoryPortOp>(
          [](auto) { return Flow::Duplex; })
      .Case<InstanceOp>([&](auto inst) {
        auto resultNo = val.cast<OpResult>().getResultNumber();
        if (inst.getPortDirection(resultNo) == Direction::Out)
          return accumulatedFlow;
        return swap();
      })
      .Case<MemOp>([&](auto op) {
        // only debug ports with RefType have source flow.
        if (val.getType().isa<RefType>())
          return Flow::Source;
        return swap();
      })
      // Anything else acts like a universal source.
      .Default([&](auto) { return accumulatedFlow; });
}

// TODO: This is doing the same walk as foldFlow.  These two functions can be
// combined and return a (flow, kind) product.
DeclKind firrtl::getDeclarationKind(Value val) {
  Operation *op = val.getDefiningOp();
  if (!op)
    return DeclKind::Port;

  return TypeSwitch<Operation *, DeclKind>(op)
      .Case<InstanceOp>([](auto) { return DeclKind::Instance; })
      .Case<SubfieldOp, SubindexOp, SubaccessOp>(
          [](auto op) { return getDeclarationKind(op.getInput()); })
      .Default([](auto) { return DeclKind::Other; });
}

size_t firrtl::getNumPorts(Operation *op) {
  if (auto module = dyn_cast<hw::HWModuleLike>(*op))
    return module.getNumPorts();
  return op->getNumResults();
}

/// Check whether an operation has a `DontTouch` annotation, or a symbol that
/// should prevent certain types of canonicalizations.
bool firrtl::hasDontTouch(Operation *op) {
  return op->getAttr(hw::InnerName::getInnerNameAttrName()) ||
         AnnotationSet(op).hasDontTouch();
}

/// Check whether a block argument ("port") or the operation defining a value
/// has a `DontTouch` annotation, or a symbol that should prevent certain types
/// of canonicalizations.
bool firrtl::hasDontTouch(Value value) {
  if (auto *op = value.getDefiningOp())
    return hasDontTouch(op);
  auto arg = value.dyn_cast<BlockArgument>();
  auto module = cast<FModuleOp>(arg.getOwner()->getParentOp());
  return (module.getPortSymbolAttr(arg.getArgNumber())) ||
         AnnotationSet::forPort(module, arg.getArgNumber()).hasDontTouch();
}

/// Get a special name to use when printing the entry block arguments of the
/// region contained by an operation in this dialect.
void getAsmBlockArgumentNamesImpl(Operation *op, mlir::Region &region,
                                  OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto *parentOp = op;
  auto *block = &region.front();
  // Check to see if the operation containing the arguments has 'firrtl.name'
  // attributes for them.  If so, use that as the name.
  auto argAttr = parentOp->getAttrOfType<ArrayAttr>("portNames");
  // Do not crash on invalid IR.
  if (!argAttr || argAttr.size() != block->getNumArguments())
    return;

  for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
    auto str = argAttr[i].cast<StringAttr>().getValue();
    if (!str.empty())
      setNameFn(block->getArgument(i), str);
  }
}

/// A forward declaration for `NameKind` attribute parser.
static ParseResult parseNameKind(OpAsmParser &parser,
                                 firrtl::NameKindEnumAttr &result);

//===----------------------------------------------------------------------===//
// CircuitOp
//===----------------------------------------------------------------------===//

void CircuitOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name, ArrayAttr annotations) {
  // Add an attribute for the name.
  result.addAttribute(builder.getStringAttr("name"), name);

  if (!annotations)
    annotations = builder.getArrayAttr({});
  result.addAttribute("annotations", annotations);

  // Create a region and a block for the body.
  Region *bodyRegion = result.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
}

// Return the main module that is the entry point of the circuit.
FModuleLike CircuitOp::getMainModule(mlir::SymbolTable *symtbl) {
  if (symtbl)
    return symtbl->lookup<FModuleLike>(getName());
  return dyn_cast_or_null<FModuleLike>(lookupSymbol(getName()));
}

static ParseResult parseCircuitOpAttrs(OpAsmParser &parser,
                                       NamedAttrList &resultAttrs) {
  auto result = parser.parseOptionalAttrDictWithKeyword(resultAttrs);
  if (!resultAttrs.get("annotations"))
    resultAttrs.append("annotations", parser.getBuilder().getArrayAttr({}));

  return result;
}

static void printCircuitOpAttrs(OpAsmPrinter &p, Operation *op,
                                DictionaryAttr attr) {
  // "name" is always elided.
  SmallVector<StringRef> elidedAttrs = {"name"};
  // Elide "annotations" if it doesn't exist or if it is empty
  auto annotationsAttr = op->getAttrOfType<ArrayAttr>("annotations");
  if (annotationsAttr.empty())
    elidedAttrs.push_back("annotations");

  p.printOptionalAttrDictWithKeyword(op->getAttrs(), elidedAttrs);
}

LogicalResult CircuitOp::verifyRegions() {
  StringRef main = getName();

  // Check that the circuit has a non-empty name.
  if (main.empty()) {
    emitOpError("must have a non-empty name");
    return failure();
  }

  mlir::SymbolTable symtbl(getOperation());

  // Check that a module matching the "main" module exists in the circuit.
  auto mainModule = getMainModule(&symtbl);
  if (!mainModule) {
    emitOpError("must contain one module that matches main name '" + main +
                "'");
    return failure();
  }

  // Check that the main module is public.
  if (!cast<hw::HWModuleLike>(*mainModule).isPublic()) {
    emitOpError("main module '" + main + "' must be public");
    return failure();
  }

  // Store a mapping of defname to either the first external module
  // that defines it or, preferentially, the first external module
  // that defines it and has no parameters.
  llvm::DenseMap<Attribute, FExtModuleOp> defnameMap;

  auto verifyExtModule = [&](FExtModuleOp extModule) -> LogicalResult {
    if (!extModule)
      return success();

    auto defname = extModule.getDefnameAttr();
    if (!defname)
      return success();

    // Check that this extmodule's defname does not conflict with
    // the symbol name of any module.
    if (auto collidingModule = symtbl.lookup<FModuleOp>(defname.getValue()))
      return extModule.emitOpError()
          .append("attribute 'defname' with value ", defname,
                  " conflicts with the name of another module in the circuit")
          .attachNote(collidingModule.getLoc())
          .append("previous module declared here");

    // Find an optional extmodule with a defname collision. Update
    // the defnameMap if this is the first extmodule with that
    // defname or if the current extmodule takes no parameters and
    // the collision does. The latter condition improves later
    // extmodule verification as checking against a parameterless
    // module is stricter.
    FExtModuleOp collidingExtModule;
    if (auto &value = defnameMap[defname]) {
      collidingExtModule = value;
      if (!value.getParameters().empty() && extModule.getParameters().empty())
        value = extModule;
    } else {
      value = extModule;
      // Go to the next extmodule if no extmodule with the same
      // defname was found.
      return success();
    }

    // Check that the number of ports is exactly the same.
    SmallVector<PortInfo> ports = extModule.getPorts();
    SmallVector<PortInfo> collidingPorts = collidingExtModule.getPorts();

    if (ports.size() != collidingPorts.size())
      return extModule.emitOpError()
          .append("with 'defname' attribute ", defname, " has ", ports.size(),
                  " ports which is different from a previously defined "
                  "extmodule with the same 'defname' which has ",
                  collidingPorts.size(), " ports")
          .attachNote(collidingExtModule.getLoc())
          .append("previous extmodule definition occurred here");

    // Check that ports match for name and type. Since parameters
    // *might* affect widths, ignore widths if either module has
    // parameters. Note that this allows for misdetections, but
    // has zero false positives.
    for (auto p : llvm::zip(ports, collidingPorts)) {
      StringAttr aName = std::get<0>(p).name, bName = std::get<1>(p).name;
      Type aType = std::get<0>(p).type, bType = std::get<1>(p).type;

      if (aName != bName)
        return extModule.emitOpError()
            .append("with 'defname' attribute ", defname,
                    " has a port with name ", aName,
                    " which does not match the name of the port in the same "
                    "position of a previously defined extmodule with the same "
                    "'defname', expected port to have name ",
                    bName)
            .attachNote(collidingExtModule.getLoc())
            .append("previous extmodule definition occurred here");

      if (!extModule.getParameters().empty() ||
          !collidingExtModule.getParameters().empty()) {
        // Compare base types as widthless, others must match.
        if (auto base = aType.dyn_cast<FIRRTLBaseType>())
          aType = base.getWidthlessType();
        if (auto base = bType.dyn_cast<FIRRTLBaseType>())
          bType = base.getWidthlessType();
      }
      if (aType != bType)
        return extModule.emitOpError()
            .append("with 'defname' attribute ", defname,
                    " has a port with name ", aName,
                    " which has a different type ", aType,
                    " which does not match the type of the port in the same "
                    "position of a previously defined extmodule with the same "
                    "'defname', expected port to have type ",
                    bType)
            .attachNote(collidingExtModule.getLoc())
            .append("previous extmodule definition occurred here");
    }
    return success();
  };

  for (auto &op : *getBodyBlock()) {
    // Verify external modules.
    if (auto extModule = dyn_cast<FExtModuleOp>(op)) {
      if (verifyExtModule(extModule).failed())
        return failure();
    }
  }

  return success();
}

Block *CircuitOp::getBodyBlock() { return &getBody().front(); }

//===----------------------------------------------------------------------===//
// FExtModuleOp and FModuleOp
//===----------------------------------------------------------------------===//

/// This function can extract information about ports from a module and an
/// extmodule.
SmallVector<PortInfo> FModuleOp::getPorts() {
  SmallVector<PortInfo> results;
  for (unsigned i = 0, e = getNumPorts(); i < e; ++i) {
    results.push_back({getPortNameAttr(i), getPortType(i), getPortDirection(i),
                       getPortSymbolAttr(i), getArgument(i).getLoc(),
                       AnnotationSet::forPort(*this, i)});
  }
  return results;
}

/// This function can extract information about ports from a module and an
/// extmodule or genmodule.
static SmallVector<PortInfo> getPorts(FModuleLike module) {
  // FExtModuleOp's don't have block arguments or locations for their ports.
  auto loc = module->getLoc();
  SmallVector<PortInfo> results;
  for (unsigned i = 0, e = getNumPorts(module); i < e; ++i) {
    results.push_back({module.getPortNameAttr(i), module.getPortType(i),
                       module.getPortDirection(i),
                       cast<hw::HWModuleLike>(*module).getPortSymbolAttr(i),
                       loc, AnnotationSet::forPort(module, i)});
  }
  return results;
}

SmallVector<PortInfo> FExtModuleOp::getPorts() {
  return ::getPorts(cast<FModuleLike>((Operation *)*this));
}

SmallVector<PortInfo> FMemModuleOp::getPorts() {
  return ::getPorts(cast<FModuleLike>((Operation *)*this));
}

// Return the port with the specified name.
BlockArgument FModuleOp::getArgument(size_t portNumber) {
  return getBodyBlock()->getArgument(portNumber);
}

/// Inserts the given ports. The insertion indices are expected to be in order.
/// Insertion occurs in-order, such that ports with the same insertion index
/// appear in the module in the same order they appeared in the list.
static void insertPorts(FModuleLike op,
                        ArrayRef<std::pair<unsigned, PortInfo>> ports,
                        Block *body = nullptr) {
  if (ports.empty())
    return;
  unsigned oldNumArgs = getNumPorts(op);
  unsigned newNumArgs = oldNumArgs + ports.size();

  // Add direction markers and names for new ports.
  SmallVector<Direction> existingDirections =
      direction::unpackAttribute(op.getPortDirectionsAttr());
  ArrayRef<Attribute> existingNames = op.getPortNames();
  ArrayRef<Attribute> existingTypes = op.getPortTypes();
  assert(existingDirections.size() == oldNumArgs);
  assert(existingNames.size() == oldNumArgs);
  assert(existingTypes.size() == oldNumArgs);

  SmallVector<Direction> newDirections;
  SmallVector<Attribute> newNames, newTypes, newAnnos, newSyms;
  newDirections.reserve(newNumArgs);
  newNames.reserve(newNumArgs);
  newTypes.reserve(newNumArgs);
  newAnnos.reserve(newNumArgs);
  newSyms.reserve(newNumArgs);

  auto emptyArray = ArrayAttr::get(op.getContext(), {});

  unsigned oldIdx = 0;
  auto migrateOldPorts = [&](unsigned untilOldIdx) {
    while (oldIdx < oldNumArgs && oldIdx < untilOldIdx) {
      newDirections.push_back(existingDirections[oldIdx]);
      newNames.push_back(existingNames[oldIdx]);
      newTypes.push_back(existingTypes[oldIdx]);
      newAnnos.push_back(op.getAnnotationsAttrForPort(oldIdx));
      newSyms.push_back(cast<hw::HWModuleLike>(*op).getPortSymbolAttr(oldIdx));
      ++oldIdx;
    }
  };
  for (auto &pair : llvm::enumerate(ports)) {
    auto idx = pair.value().first;
    auto &port = pair.value().second;
    auto newIdx = pair.index();
    migrateOldPorts(idx);
    newDirections.push_back(port.direction);
    newNames.push_back(port.name);
    newTypes.push_back(TypeAttr::get(port.type));
    auto annos = port.annotations.getArrayAttr();
    newAnnos.push_back(annos ? annos : emptyArray);
    newSyms.push_back(port.sym);
    if (body) {
      // Block arguments are inserted one at a time, so for each argument we
      // insert we have to increase the index by 1.
      body->insertArgument(idx + newIdx, port.type, port.loc);
    }
  }
  migrateOldPorts(oldNumArgs);

  // The lack of *any* port annotations is represented by an empty
  // `portAnnotations` array as a shorthand.
  if (llvm::all_of(newAnnos, [](Attribute attr) {
        return attr.cast<ArrayAttr>().empty();
      }))
    newAnnos.clear();

  // Apply these changed markers.
  op->setAttr("portDirections",
              direction::packAttribute(op.getContext(), newDirections));
  op->setAttr("portNames", ArrayAttr::get(op.getContext(), newNames));
  op->setAttr("portTypes", ArrayAttr::get(op.getContext(), newTypes));
  op->setAttr("portAnnotations", ArrayAttr::get(op.getContext(), newAnnos));
  op.setPortSymbols(newSyms);
}

/// Erases the ports that have their corresponding bit set in `portIndices`.
static void erasePorts(FModuleLike op, const llvm::BitVector &portIndices) {
  if (portIndices.none())
    return;

  // Drop the direction markers for dead ports.
  SmallVector<Direction> portDirections =
      direction::unpackAttribute(op.getPortDirectionsAttr());
  ArrayRef<Attribute> portNames = op.getPortNames();
  ArrayRef<Attribute> portTypes = op.getPortTypes();
  ArrayRef<Attribute> portAnnos = op.getPortAnnotations();
  ArrayRef<Attribute> portSyms = op.getPortSymbols();
  auto numPorts = getNumPorts(op);
  (void)numPorts;
  assert(portDirections.size() == numPorts);
  assert(portNames.size() == numPorts);
  assert(portAnnos.size() == numPorts || portAnnos.empty());
  assert(portTypes.size() == numPorts);
  assert(portSyms.size() == numPorts || portSyms.empty());

  SmallVector<Direction> newPortDirections =
      removeElementsAtIndices<Direction>(portDirections, portIndices);
  SmallVector<Attribute> newPortNames, newPortTypes, newPortAnnos, newPortSyms;
  newPortNames = removeElementsAtIndices(portNames, portIndices);
  newPortTypes = removeElementsAtIndices(portTypes, portIndices);
  newPortAnnos = removeElementsAtIndices(portAnnos, portIndices);
  newPortSyms = removeElementsAtIndices(portSyms, portIndices);
  op->setAttr("portDirections",
              direction::packAttribute(op.getContext(), newPortDirections));
  op->setAttr("portNames", ArrayAttr::get(op.getContext(), newPortNames));
  op->setAttr("portAnnotations", ArrayAttr::get(op.getContext(), newPortAnnos));
  op->setAttr("portTypes", ArrayAttr::get(op.getContext(), newPortTypes));
  FModuleLike::fixupPortSymsArray(newPortSyms, op.getContext());
  op->setAttr("portSyms", ArrayAttr::get(op.getContext(), newPortSyms));
}

void FExtModuleOp::erasePorts(const llvm::BitVector &portIndices) {
  return ::erasePorts(cast<FModuleLike>((Operation *)*this), portIndices);
}

void FMemModuleOp::erasePorts(const llvm::BitVector &portIndices) {
  return ::erasePorts(cast<FModuleLike>((Operation *)*this), portIndices);
}

/// Erases the ports that have their corresponding bit set in `portIndices`.
void FModuleOp::erasePorts(const llvm::BitVector &portIndices) {
  ::erasePorts(cast<FModuleLike>((Operation *)*this), portIndices);
  // Erase the block arguments.
  getBodyBlock()->eraseArguments(portIndices);
}

/// Inserts the given ports. The insertion indices are expected to be in order.
/// Insertion occurs in-order, such that ports with the same insertion index
/// appear in the module in the same order they appeared in the list.
void FModuleOp::insertPorts(ArrayRef<std::pair<unsigned, PortInfo>> ports) {
  Block *body = getBodyBlock();
  ::insertPorts(cast<FModuleLike>((Operation *)*this), ports, body);
}

void FExtModuleOp::insertPorts(ArrayRef<std::pair<unsigned, PortInfo>> ports) {
  ::insertPorts(cast<FModuleLike>((Operation *)*this), ports);
}

/// Inserts the given ports. The insertion indices are expected to be in order.
/// Insertion occurs in-order, such that ports with the same insertion index
/// appear in the module in the same order they appeared in the list.
void FMemModuleOp::insertPorts(ArrayRef<std::pair<unsigned, PortInfo>> ports) {
  ::insertPorts(cast<FModuleLike>((Operation *)*this), ports);
}

static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<PortInfo> ports,
                        ArrayAttr annotations) {
  // Add an attribute for the name.
  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  // Record the names of the arguments if present.
  SmallVector<Direction, 4> portDirections;
  SmallVector<Attribute, 4> portNames;
  SmallVector<Attribute, 4> portTypes;
  SmallVector<Attribute, 4> portAnnotations;
  SmallVector<Attribute, 4> portSyms;
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    portDirections.push_back(ports[i].direction);
    portNames.push_back(ports[i].name);
    portTypes.push_back(TypeAttr::get(ports[i].type));
    portAnnotations.push_back(ports[i].annotations.getArrayAttr());
    portSyms.push_back(ports[i].sym);
  }

  // The lack of *any* port annotations is represented by an empty
  // `portAnnotations` array as a shorthand.
  if (llvm::all_of(portAnnotations, [](Attribute attr) {
        return attr.cast<ArrayAttr>().empty();
      }))
    portAnnotations.clear();

  FModuleLike::fixupPortSymsArray(portSyms, builder.getContext());
  // Both attributes are added, even if the module has no ports.
  result.addAttribute(
      "portDirections",
      direction::packAttribute(builder.getContext(), portDirections));
  result.addAttribute("portNames", builder.getArrayAttr(portNames));
  result.addAttribute("portTypes", builder.getArrayAttr(portTypes));
  result.addAttribute("portAnnotations", builder.getArrayAttr(portAnnotations));
  result.addAttribute("portSyms", builder.getArrayAttr(portSyms));

  if (!annotations)
    annotations = builder.getArrayAttr({});
  result.addAttribute("annotations", annotations);

  result.addRegion();
}

void FModuleOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name, ArrayRef<PortInfo> ports,
                      ArrayAttr annotations) {
  buildModule(builder, result, name, ports, annotations);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports)
    body->addArgument(elt.type, elt.loc);
}

void FExtModuleOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name, ArrayRef<PortInfo> ports,
                         StringRef defnameAttr, ArrayAttr annotations,
                         ArrayAttr parameters) {
  buildModule(builder, result, name, ports, annotations);
  if (!defnameAttr.empty())
    result.addAttribute("defname", builder.getStringAttr(defnameAttr));
  if (!parameters)
    result.addAttribute("parameters", builder.getArrayAttr({}));
}

void FMemModuleOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name, ArrayRef<PortInfo> ports,
                         uint32_t numReadPorts, uint32_t numWritePorts,
                         uint32_t numReadWritePorts, uint32_t dataWidth,
                         uint32_t maskBits, uint32_t readLatency,
                         uint32_t writeLatency, uint64_t depth,
                         ArrayAttr annotations) {
  auto *context = builder.getContext();
  buildModule(builder, result, name, ports, annotations);
  auto ui32Type = IntegerType::get(context, 32, IntegerType::Unsigned);
  auto ui64Type = IntegerType::get(context, 64, IntegerType::Unsigned);
  result.addAttribute("numReadPorts", IntegerAttr::get(ui32Type, numReadPorts));
  result.addAttribute("numWritePorts",
                      IntegerAttr::get(ui32Type, numWritePorts));
  result.addAttribute("numReadWritePorts",
                      IntegerAttr::get(ui32Type, numReadWritePorts));
  result.addAttribute("dataWidth", IntegerAttr::get(ui32Type, dataWidth));
  result.addAttribute("maskBits", IntegerAttr::get(ui32Type, maskBits));
  result.addAttribute("readLatency", IntegerAttr::get(ui32Type, readLatency));
  result.addAttribute("writeLatency", IntegerAttr::get(ui32Type, writeLatency));
  result.addAttribute("depth", IntegerAttr::get(ui64Type, depth));
  result.addAttribute("extraPorts", ArrayAttr::get(context, {}));
}

/// Print a list of module ports in the following form:
///   in x: !firrtl.uint<1> [{class = "DontTouch}], out "_port": !firrtl.uint<2>
///
/// When there is no block specified, the port names print as MLIR identifiers,
/// wrapping in quotes if not legal to print as-is. When there is no block
/// specified, this function always return false, indicating that there was no
/// issue printing port names.
///
/// If there is a block specified, then port names will be printed as SSA
/// values.  If there is a reason the printed SSA values can't match the true
/// port name, then this function will return true.  When this happens, the
/// caller should print the port names as a part of the `attr-dict`.
static bool printModulePorts(OpAsmPrinter &p, Block *block,
                             ArrayRef<Direction> portDirections,
                             ArrayRef<Attribute> portNames,
                             ArrayRef<Attribute> portTypes,
                             ArrayRef<Attribute> portAnnotations,
                             ArrayRef<Attribute> portSyms) {
  // When printing port names as SSA values, we can fail to print them
  // identically.
  bool printedNamesDontMatch = false;

  // If we are printing the ports as block arguments the op must have a first
  // block.
  SmallString<32> resultNameStr;
  p << '(';
  for (unsigned i = 0, e = portTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    // Print the port direction.
    p << portDirections[i] << " ";

    // Print the port name.  If there is a valid block, we print it as a block
    // argument.
    if (block) {
      // Get the printed format for the argument name.
      resultNameStr.clear();
      llvm::raw_svector_ostream tmpStream(resultNameStr);
      p.printOperand(block->getArgument(i), tmpStream);
      // If the name wasn't printable in a way that agreed with portName, make
      // sure to print out an explicit portNames attribute.
      auto portName = portNames[i].cast<StringAttr>().getValue();
      if (!portName.empty() && tmpStream.str().drop_front() != portName)
        printedNamesDontMatch = true;
      p << tmpStream.str();
    } else {
      p.printKeywordOrString(portNames[i].cast<StringAttr>().getValue());
    }

    // Print the port type.
    p << ": ";
    auto portType = portTypes[i].cast<TypeAttr>().getValue();
    p.printType(portType);

    // Print the optional port symbol.
    if (!portSyms.empty()) {
      if (!portSyms[i].cast<hw::InnerSymAttr>().empty()) {
        p << " sym ";
        portSyms[i].cast<hw::InnerSymAttr>().print(p);
      }
    }

    // Print the port specific annotations. The port annotations array will be
    // empty if there are none.
    if (!portAnnotations.empty() &&
        !portAnnotations[i].cast<ArrayAttr>().empty()) {
      p << " ";
      p.printAttribute(portAnnotations[i]);
    }
  }

  p << ')';
  return printedNamesDontMatch;
}

/// Parse a list of module ports.  If port names are SSA identifiers, then this
/// will populate `entryArgs`.
static ParseResult
parseModulePorts(OpAsmParser &parser, bool hasSSAIdentifiers,
                 SmallVectorImpl<OpAsmParser::Argument> &entryArgs,
                 SmallVectorImpl<Direction> &portDirections,
                 SmallVectorImpl<Attribute> &portNames,
                 SmallVectorImpl<Attribute> &portTypes,
                 SmallVectorImpl<Attribute> &portAnnotations,
                 SmallVectorImpl<Attribute> &portSyms) {
  auto *context = parser.getContext();

  auto parseArgument = [&]() -> ParseResult {
    // Parse port direction.
    if (succeeded(parser.parseOptionalKeyword("out")))
      portDirections.push_back(Direction::Out);
    else if (succeeded(parser.parseKeyword("in", "or 'out'")))
      portDirections.push_back(Direction::In);
    else
      return failure();

    // Parse the port name.
    if (hasSSAIdentifiers) {
      OpAsmParser::Argument arg;
      if (parser.parseArgument(arg))
        return failure();
      entryArgs.push_back(arg);
      // The name of an argument is of the form "%42" or "%id", and since
      // parsing succeeded, we know it always has one character.
      assert(arg.ssaName.name.size() > 1 && arg.ssaName.name[0] == '%' &&
             "Unknown MLIR name");
      if (isdigit(arg.ssaName.name[1]))
        portNames.push_back(StringAttr::get(context, ""));
      else
        portNames.push_back(
            StringAttr::get(context, arg.ssaName.name.drop_front()));
    } else {
      std::string portName;
      if (parser.parseKeywordOrString(&portName))
        return failure();
      portNames.push_back(StringAttr::get(context, portName));
    }

    // Parse the port type.
    Type portType;
    if (parser.parseColonType(portType))
      return failure();
    portTypes.push_back(TypeAttr::get(portType));

    if (hasSSAIdentifiers)
      entryArgs.back().type = portType;

    // Parse the optional port symbol.
    hw::InnerSymAttr innerSymAttr;
    if (succeeded(parser.parseOptionalKeyword("sym"))) {
      NamedAttrList dummyAttrs;
      if (parser.parseCustomAttributeWithFallback(
              innerSymAttr, ::mlir::Type{},
              hw::InnerSymbolTable::getInnerSymbolAttrName(), dummyAttrs)) {
        return ::mlir::failure();
      }
    }
    portSyms.push_back(innerSymAttr);

    // Parse the port annotations.
    ArrayAttr annos;
    auto parseResult = parser.parseOptionalAttribute(annos);
    if (!parseResult.has_value())
      annos = parser.getBuilder().getArrayAttr({});
    else if (failed(*parseResult))
      return failure();
    portAnnotations.push_back(annos);

    return success();
  };

  // Parse all ports.
  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseArgument);
}

/// Print a paramter list for a module or instance.
static void printParameterList(ArrayAttr parameters, OpAsmPrinter &p) {
  if (!parameters || parameters.empty())
    return;

  p << '<';
  llvm::interleaveComma(parameters, p, [&](Attribute param) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    p << paramAttr.getName().getValue() << ": " << paramAttr.getType();
    if (auto value = paramAttr.getValue()) {
      p << " = ";
      p.printAttributeWithoutType(value);
    }
  });
  p << '>';
}

static void printFModuleLikeOp(OpAsmPrinter &p, FModuleLike op) {
  p << " ";

  // Print the visibility of the module.
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';

  // Print the operation and the function name.
  p.printSymbolName(op.moduleName());

  // Print the parameter list (if non-empty).
  printParameterList(op->getAttrOfType<ArrayAttr>("parameters"), p);

  // Both modules and external modules have a body, but it is always empty for
  // external modules.
  Block *body = nullptr;
  if (!op->getRegion(0).empty())
    body = &op->getRegion(0).front();

  auto portDirections = direction::unpackAttribute(op.getPortDirectionsAttr());

  auto needPortNamesAttr = printModulePorts(
      p, body, portDirections, op.getPortNames(), op.getPortTypes(),
      op.getPortAnnotations(), op.getPortSymbols());

  SmallVector<StringRef, 4> omittedAttrs = {
      "sym_name", "portDirections", "portTypes",       "portAnnotations",
      "portSyms", "parameters",     visibilityAttrName};

  // We can omit the portNames if they were able to be printed as properly as
  // block arguments.
  if (!needPortNamesAttr)
    omittedAttrs.push_back("portNames");

  // If there are no annotations we can omit the empty array.
  if (op->getAttrOfType<ArrayAttr>("annotations").empty())
    omittedAttrs.push_back("annotations");

  // If there are no internal paths attributes we can omit the empty array.
  if (op->hasAttr("internalPaths")) {
    if (auto paths = op->getAttrOfType<ArrayAttr>("internalPaths"))
      if (paths.empty())
        omittedAttrs.push_back("internalPaths");
  }

  p.printOptionalAttrDictWithKeyword(op->getAttrs(), omittedAttrs);
}

void FExtModuleOp::print(OpAsmPrinter &p) { printFModuleLikeOp(p, *this); }

void FMemModuleOp::print(OpAsmPrinter &p) { printFModuleLikeOp(p, *this); }

void FModuleOp::print(OpAsmPrinter &p) {
  printFModuleLikeOp(p, *this);

  // Print the body if this is not an external function. Since this block does
  // not have terminators, printing the terminator actually just prints the last
  // operation.
  Region &fbody = getBody();
  if (!fbody.empty()) {
    p << " ";
    p.printRegion(fbody, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

/// Parse an parameter list if present.
/// module-parameter-list ::= `<` parameter-decl (`,` parameter-decl)* `>`
/// parameter-decl ::= identifier `:` type
/// parameter-decl ::= identifier `:` type `=` attribute
///
static ParseResult
parseOptionalParameters(OpAsmParser &parser,
                        SmallVectorImpl<Attribute> &parameters) {

  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::OptionalLessGreater, [&]() {
        std::string name;
        Type type;
        Attribute value;

        if (parser.parseKeywordOrString(&name) || parser.parseColonType(type))
          return failure();

        // Parse the default value if present.
        if (succeeded(parser.parseOptionalEqual())) {
          if (parser.parseAttribute(value, type))
            return failure();
        }

        auto &builder = parser.getBuilder();
        parameters.push_back(ParamDeclAttr::get(
            builder.getContext(), builder.getStringAttr(name), type, value));
        return success();
      });
}

static ParseResult parseFModuleLikeOp(OpAsmParser &parser,
                                      OperationState &result,
                                      bool hasSSAIdentifiers) {
  auto *context = result.getContext();
  auto &builder = parser.getBuilder();

  // Parse the visibility attribute.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse optional parameters.
  SmallVector<Attribute, 4> parameters;
  if (parseOptionalParameters(parser, parameters))
    return failure();
  result.addAttribute("parameters", builder.getArrayAttr(parameters));

  // Parse the module ports.
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<Direction, 4> portDirections;
  SmallVector<Attribute, 4> portNames;
  SmallVector<Attribute, 4> portTypes;
  SmallVector<Attribute, 4> portAnnotations;
  SmallVector<Attribute, 4> portSyms;
  if (parseModulePorts(parser, hasSSAIdentifiers, entryArgs, portDirections,
                       portNames, portTypes, portAnnotations, portSyms))
    return failure();

  // If module attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  assert(portNames.size() == portTypes.size());

  // Record the argument and result types as an attribute.  This is necessary
  // for external modules.

  // Add port directions.
  if (!result.attributes.get("portDirections"))
    result.addAttribute("portDirections",
                        direction::packAttribute(context, portDirections));

  // Add port names.
  if (!result.attributes.get("portNames")) {
    result.addAttribute("portNames", builder.getArrayAttr(portNames));
  }

  // Add the port types.
  if (!result.attributes.get("portTypes"))
    result.addAttribute("portTypes", ArrayAttr::get(context, portTypes));

  // Add the port annotations.
  if (!result.attributes.get("portAnnotations")) {
    // If there are no portAnnotations, don't add the attribute.
    if (llvm::any_of(portAnnotations, [&](Attribute anno) {
          return !anno.cast<ArrayAttr>().empty();
        }))
      result.addAttribute("portAnnotations",
                          ArrayAttr::get(context, portAnnotations));
  }

  // Add port symbols.
  if (!result.attributes.get("portSyms")) {
    FModuleLike::fixupPortSymsArray(portSyms, builder.getContext());
    result.addAttribute("portSyms", builder.getArrayAttr(portSyms));
  }

  // The annotations attribute is always present, but not printed when empty.
  if (!result.attributes.get("annotations"))
    result.addAttribute("annotations", builder.getArrayAttr({}));

  // The portAnnotations attribute is always present, but not printed when
  // empty.
  if (!result.attributes.get("portAnnotations"))
    result.addAttribute("portAnnotations", builder.getArrayAttr({}));

  // Parse the optional function body.
  auto *body = result.addRegion();

  if (hasSSAIdentifiers) {
    if (parser.parseRegion(*body, entryArgs))
      return failure();
    if (body->empty())
      body->push_back(new Block());
  }
  return success();
}

ParseResult FModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFModuleLikeOp(parser, result, /*hasSSAIdentifiers=*/true);
}

ParseResult FExtModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFModuleLikeOp(parser, result, /*hasSSAIdentifiers=*/false);
}

ParseResult FMemModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFModuleLikeOp(parser, result, /*hasSSAIdentifiers=*/false);
}

LogicalResult FExtModuleOp::verify() {
  auto params = getParameters();
  if (params.empty())
    return success();

  auto checkParmValue = [&](Attribute elt) -> bool {
    auto param = elt.cast<ParamDeclAttr>();
    auto value = param.getValue();
    if (value.isa<IntegerAttr>() || value.isa<StringAttr>() ||
        value.isa<FloatAttr>())
      return true;
    emitError() << "has unknown extmodule parameter value '"
                << param.getName().getValue() << "' = " << value;
    return false;
  };

  if (!llvm::all_of(params, checkParmValue))
    return failure();

  return success();
}

void FModuleOp::getAsmBlockArgumentNames(mlir::Region &region,
                                         mlir::OpAsmSetValueNameFn setNameFn) {
  getAsmBlockArgumentNamesImpl(getOperation(), region, setNameFn);
}

void FExtModuleOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  getAsmBlockArgumentNamesImpl(getOperation(), region, setNameFn);
}

void FMemModuleOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  getAsmBlockArgumentNamesImpl(getOperation(), region, setNameFn);
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
Operation *InstanceOp::getReferencedModule() {
  auto circuit = (*this)->getParentOfType<CircuitOp>();
  if (!circuit)
    return nullptr;

  return circuit.lookupSymbol<FModuleLike>(getModuleNameAttr());
}

FModuleLike InstanceOp::getReferencedModule(SymbolTable &symbolTable) {
  return symbolTable.lookup<FModuleLike>(
      getModuleNameAttr().getLeafReference());
}

void InstanceOp::build(OpBuilder &builder, OperationState &result,
                       TypeRange resultTypes, StringRef moduleName,
                       StringRef name, NameKindEnum nameKind,
                       ArrayRef<Direction> portDirections,
                       ArrayRef<Attribute> portNames,
                       ArrayRef<Attribute> annotations,
                       ArrayRef<Attribute> portAnnotations, bool lowerToBind,
                       StringAttr innerSym) {
  build(builder, result, resultTypes, moduleName, name, nameKind,
        portDirections, portNames, annotations, portAnnotations, lowerToBind,
        innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr());
}

void InstanceOp::build(OpBuilder &builder, OperationState &result,
                       TypeRange resultTypes, StringRef moduleName,
                       StringRef name, NameKindEnum nameKind,
                       ArrayRef<Direction> portDirections,
                       ArrayRef<Attribute> portNames,
                       ArrayRef<Attribute> annotations,
                       ArrayRef<Attribute> portAnnotations, bool lowerToBind,
                       hw::InnerSymAttr innerSym) {
  result.addTypes(resultTypes);
  result.addAttribute("moduleName",
                      SymbolRefAttr::get(builder.getContext(), moduleName));
  result.addAttribute("name", builder.getStringAttr(name));
  result.addAttribute(
      "portDirections",
      direction::packAttribute(builder.getContext(), portDirections));
  result.addAttribute("portNames", builder.getArrayAttr(portNames));
  result.addAttribute("annotations", builder.getArrayAttr(annotations));
  if (lowerToBind)
    result.addAttribute("lowerToBind", builder.getUnitAttr());
  if (innerSym)
    result.addAttribute("inner_sym", innerSym);
  result.addAttribute("nameKind",
                      NameKindEnumAttr::get(builder.getContext(), nameKind));

  if (portAnnotations.empty()) {
    SmallVector<Attribute, 16> portAnnotationsVec(resultTypes.size(),
                                                  builder.getArrayAttr({}));
    result.addAttribute("portAnnotations",
                        builder.getArrayAttr(portAnnotationsVec));
  } else {
    assert(portAnnotations.size() == resultTypes.size());
    result.addAttribute("portAnnotations",
                        builder.getArrayAttr(portAnnotations));
  }
}

void InstanceOp::build(OpBuilder &builder, OperationState &result,
                       FModuleLike module, StringRef name,
                       NameKindEnum nameKind, ArrayRef<Attribute> annotations,
                       ArrayRef<Attribute> portAnnotations, bool lowerToBind,
                       StringAttr innerSym) {

  // Gather the result types.
  SmallVector<Type> resultTypes;
  resultTypes.reserve(getNumPorts(module));
  llvm::transform(
      module.getPortTypes(), std::back_inserter(resultTypes),
      [](Attribute typeAttr) { return typeAttr.cast<TypeAttr>().getValue(); });

  // Create the port annotations.
  ArrayAttr portAnnotationsAttr;
  if (portAnnotations.empty()) {
    portAnnotationsAttr = builder.getArrayAttr(SmallVector<Attribute, 16>(
        resultTypes.size(), builder.getArrayAttr({})));
  } else {
    portAnnotationsAttr = builder.getArrayAttr(portAnnotations);
  }

  return build(
      builder, result, resultTypes,
      SymbolRefAttr::get(builder.getContext(), module.moduleNameAttr()),
      builder.getStringAttr(name),
      NameKindEnumAttr::get(builder.getContext(), nameKind),
      module.getPortDirectionsAttr(), module.getPortNamesAttr(),
      builder.getArrayAttr(annotations), portAnnotationsAttr,
      lowerToBind ? builder.getUnitAttr() : UnitAttr(),
      innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr());
}

/// Builds a new `InstanceOp` with the ports listed in `portIndices` erased, and
/// updates any users of the remaining ports to point at the new instance.
InstanceOp InstanceOp::erasePorts(OpBuilder &builder,
                                  const llvm::BitVector &portIndices) {
  assert(portIndices.size() >= getNumResults() &&
         "portIndices is not at least as large as getNumResults()");

  if (portIndices.none())
    return *this;

  SmallVector<Type> newResultTypes = removeElementsAtIndices<Type>(
      SmallVector<Type>(result_type_begin(), result_type_end()), portIndices);
  SmallVector<Direction> newPortDirections = removeElementsAtIndices<Direction>(
      direction::unpackAttribute(getPortDirectionsAttr()), portIndices);
  SmallVector<Attribute> newPortNames =
      removeElementsAtIndices(getPortNames().getValue(), portIndices);
  SmallVector<Attribute> newPortAnnotations =
      removeElementsAtIndices(getPortAnnotations().getValue(), portIndices);

  auto newOp = builder.create<InstanceOp>(
      getLoc(), newResultTypes, getModuleName(), getName(), getNameKind(),
      newPortDirections, newPortNames, getAnnotations().getValue(),
      newPortAnnotations, getLowerToBind(), getInnerSymAttr());

  for (unsigned oldIdx = 0, newIdx = 0, numOldPorts = getNumResults();
       oldIdx != numOldPorts; ++oldIdx) {
    if (portIndices.test(oldIdx)) {
      assert(getResult(oldIdx).use_empty() && "removed instance port has uses");
      continue;
    }
    getResult(oldIdx).replaceAllUsesWith(newOp.getResult(newIdx));
    ++newIdx;
  }

  // Compy over "output_file" information so that this is not lost when ports
  // are erased.
  //
  // TODO: Other attributes may need to be copied over.
  if (auto outputFile = (*this)->getAttr("output_file"))
    newOp->setAttr("output_file", outputFile);

  return newOp;
}

ArrayAttr InstanceOp::getPortAnnotation(unsigned portIdx) {
  assert(portIdx < getNumResults() &&
         "index should be smaller than result number");
  return getPortAnnotations()[portIdx].cast<ArrayAttr>();
}

void InstanceOp::setAllPortAnnotations(ArrayRef<Attribute> annotations) {
  assert(annotations.size() == getNumResults() &&
         "number of annotations is not equal to result number");
  (*this)->setAttr("portAnnotations",
                   ArrayAttr::get(getContext(), annotations));
}

InstanceOp
InstanceOp::cloneAndInsertPorts(ArrayRef<std::pair<unsigned, PortInfo>> ports) {
  auto portSize = ports.size();
  auto newPortCount = getNumResults() + portSize;
  SmallVector<Direction> newPortDirections;
  newPortDirections.reserve(newPortCount);
  SmallVector<Attribute> newPortNames;
  newPortNames.reserve(newPortCount);
  SmallVector<Type> newPortTypes;
  newPortTypes.reserve(newPortCount);
  SmallVector<Attribute> newPortAnnos;
  newPortAnnos.reserve(newPortCount);

  unsigned oldIndex = 0;
  unsigned newIndex = 0;
  while (oldIndex + newIndex < newPortCount) {
    // Check if we should insert a port here.
    if (newIndex < portSize && ports[newIndex].first == oldIndex) {
      auto &newPort = ports[newIndex].second;
      newPortDirections.push_back(newPort.direction);
      newPortNames.push_back(newPort.name);
      newPortTypes.push_back(newPort.type);
      newPortAnnos.push_back(newPort.annotations.getArrayAttr());
      ++newIndex;
    } else {
      // Copy the next old port.
      newPortDirections.push_back(getPortDirection(oldIndex));
      newPortNames.push_back(getPortName(oldIndex));
      newPortTypes.push_back(getType(oldIndex));
      newPortAnnos.push_back(getPortAnnotation(oldIndex));
      ++oldIndex;
    }
  }

  // Create a new instance op with the reset inserted.
  return OpBuilder(*this).create<InstanceOp>(
      getLoc(), newPortTypes, getModuleName(), getName(), getNameKind(),
      newPortDirections, newPortNames, getAnnotations().getValue(),
      newPortAnnos, getLowerToBind(), getInnerSymAttr());
}

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto module = (*this)->getParentOfType<FModuleOp>();
  auto referencedModule = symbolTable.lookupNearestSymbolFrom<FModuleLike>(
      *this, getModuleNameAttr());
  if (!referencedModule) {
    return emitOpError("invalid symbol reference");
  }

  // Check that this instance doesn't recursively instantiate its wrapping
  // module.
  if (referencedModule == module) {
    auto diag = emitOpError()
                << "is a recursive instantiation of its containing module";
    return diag.attachNote(module.getLoc())
           << "containing module declared here";
  }

  // Small helper add a note to the original declaration.
  auto emitNote = [&](InFlightDiagnostic &&diag) -> InFlightDiagnostic && {
    diag.attachNote(referencedModule->getLoc())
        << "original module declared here";
    return std::move(diag);
  };

  // Check that all the attribute arrays are the right length up front.  This
  // lets us safely use the port name in error messages below.
  size_t numResults = getNumResults();
  size_t numExpected = getNumPorts(referencedModule);
  if (numResults != numExpected) {
    return emitNote(emitOpError() << "has a wrong number of results; expected "
                                  << numExpected << " but got " << numResults);
  }
  if (getPortDirections().getBitWidth() != numExpected)
    return emitNote(emitOpError("the number of port directions should be "
                                "equal to the number of results"));
  if (getPortNames().size() != numExpected)
    return emitNote(emitOpError("the number of port names should be "
                                "equal to the number of results"));
  if (getPortAnnotations().size() != numExpected)
    return emitNote(emitOpError("the number of result annotations should be "
                                "equal to the number of results"));

  // Check that the port names match the referenced module.
  if (getPortNamesAttr() != referencedModule.getPortNamesAttr()) {
    // We know there is an error, try to figure out whats wrong.
    auto instanceNames = getPortNames();
    auto moduleNames = referencedModule.getPortNamesAttr();
    // First compare the sizes:
    if (instanceNames.size() != moduleNames.size()) {
      return emitNote(emitOpError()
                      << "has a wrong number of directions; expected "
                      << moduleNames.size() << " but got "
                      << instanceNames.size());
    }
    // Next check the values:
    for (size_t i = 0; i != numResults; ++i) {
      if (instanceNames[i] != moduleNames[i]) {
        return emitNote(emitOpError()
                        << "name for port " << i << " must be "
                        << moduleNames[i] << ", but got " << instanceNames[i]);
      }
    }
    llvm_unreachable("should have found something wrong");
  }

  // Check that the types match.
  for (size_t i = 0; i != numResults; i++) {
    auto resultType = getResult(i).getType();
    auto expectedType = referencedModule.getPortType(i);
    if (resultType != expectedType) {
      return emitNote(emitOpError()
                      << "result type for " << getPortName(i) << " must be "
                      << expectedType << ", but got " << resultType);
    }
  }

  // Check that the port directions are consistent with the referenced module's.
  if (getPortDirectionsAttr() != referencedModule.getPortDirectionsAttr()) {
    // We know there is an error, try to figure out whats wrong.
    auto instanceDirectionAttr = getPortDirectionsAttr();
    auto moduleDirectionAttr = referencedModule.getPortDirectionsAttr();
    // First compare the sizes:
    auto expectedWidth = moduleDirectionAttr.getValue().getBitWidth();
    auto actualWidth = instanceDirectionAttr.getValue().getBitWidth();
    if (expectedWidth != actualWidth) {
      return emitNote(emitOpError()
                      << "has a wrong number of directions; expected "
                      << expectedWidth << " but got " << actualWidth);
    }
    // Next check the values.
    auto instanceDirs = direction::unpackAttribute(instanceDirectionAttr);
    auto moduleDirs = direction::unpackAttribute(moduleDirectionAttr);
    for (size_t i = 0; i != numResults; ++i) {
      if (instanceDirs[i] != moduleDirs[i]) {
        return emitNote(emitOpError()
                        << "direction for " << getPortName(i) << " must be \""
                        << direction::toString(moduleDirs[i])
                        << "\", but got \""
                        << direction::toString(instanceDirs[i]) << "\"");
      }
    }
    llvm_unreachable("should have found something wrong");
  }

  return success();
}

StringRef InstanceOp::instanceName() { return getName(); }

StringAttr InstanceOp::instanceNameAttr() { return getNameAttr(); }

void InstanceOp::print(OpAsmPrinter &p) {
  // Print the instance name.
  p << " ";
  p.printKeywordOrString(getName());
  if (auto attr = getInnerSymAttr()) {
    p << " sym ";
    p.printSymbolName(attr.getSymName());
  }
  if (getNameKindAttr().getValue() != NameKindEnum::DroppableName)
    p << ' ' << stringifyNameKindEnum(getNameKindAttr().getValue());
  p << " ";

  // Print the attr-dict.
  SmallVector<StringRef, 4> omittedAttrs = {"moduleName",     "name",
                                            "portDirections", "portNames",
                                            "portTypes",      "portAnnotations",
                                            "inner_sym",      "nameKind"};
  if (getAnnotations().empty())
    omittedAttrs.push_back("annotations");
  p.printOptionalAttrDict((*this)->getAttrs(), omittedAttrs);

  // Print the module name.
  p << " ";
  p.printSymbolName(getModuleName());

  // Collect all the result types as TypeAttrs for printing.
  SmallVector<Attribute> portTypes;
  portTypes.reserve(getNumResults());
  llvm::transform(getResultTypes(), std::back_inserter(portTypes),
                  &TypeAttr::get);
  auto portDirections = direction::unpackAttribute(getPortDirectionsAttr());
  printModulePorts(p, /*block=*/nullptr, portDirections,
                   getPortNames().getValue(), portTypes,
                   getPortAnnotations().getValue(), {});
}

ParseResult InstanceOp::parse(OpAsmParser &parser, OperationState &result) {
  auto *context = parser.getContext();
  auto &resultAttrs = result.attributes;

  std::string name;
  hw::InnerSymAttr innerSymAttr;
  FlatSymbolRefAttr moduleName;
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<Direction, 4> portDirections;
  SmallVector<Attribute, 4> portNames;
  SmallVector<Attribute, 4> portTypes;
  SmallVector<Attribute, 4> portAnnotations;
  SmallVector<Attribute, 4> portSyms;
  NameKindEnumAttr nameKind;

  if (parser.parseKeywordOrString(&name))
    return failure();
  if (succeeded(parser.parseOptionalKeyword("sym"))) {
    if (parser.parseCustomAttributeWithFallback(
            innerSymAttr, ::mlir::Type{},
            hw::InnerSymbolTable::getInnerSymbolAttrName(),
            result.attributes)) {
      return ::mlir::failure();
    }
  }
  if (parseNameKind(parser, nameKind) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(moduleName, "moduleName", resultAttrs) ||
      parseModulePorts(parser, /*hasSSAIdentifiers=*/false, entryArgs,
                       portDirections, portNames, portTypes, portAnnotations,
                       portSyms))
    return failure();

  // Add the attributes. We let attributes defined in the attr-dict override
  // attributes parsed out of the module signature.
  if (!resultAttrs.get("moduleName"))
    result.addAttribute("moduleName", moduleName);
  if (!resultAttrs.get("name"))
    result.addAttribute("name", StringAttr::get(context, name));
  result.addAttribute("nameKind", nameKind);
  if (!resultAttrs.get("portDirections"))
    result.addAttribute("portDirections",
                        direction::packAttribute(context, portDirections));
  if (!resultAttrs.get("portNames"))
    result.addAttribute("portNames", ArrayAttr::get(context, portNames));
  if (!resultAttrs.get("portAnnotations"))
    result.addAttribute("portAnnotations",
                        ArrayAttr::get(context, portAnnotations));

  // Annotations and LowerToBind are omitted in the printed format if they are
  // empty and false, respectively.
  if (!resultAttrs.get("annotations"))
    resultAttrs.append("annotations", parser.getBuilder().getArrayAttr({}));

  // Add result types.
  result.types.reserve(portTypes.size());
  llvm::transform(
      portTypes, std::back_inserter(result.types),
      [](Attribute typeAttr) { return typeAttr.cast<TypeAttr>().getValue(); });

  return success();
}

void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  StringRef base = getName();
  if (base.empty())
    base = "inst";

  for (size_t i = 0, e = (*this)->getNumResults(); i != e; ++i) {
    setNameFn(getResult(i), (base + "_" + getPortNameStr(i)).str());
  }
}

void MemOp::build(OpBuilder &builder, OperationState &result,
                  TypeRange resultTypes, uint32_t readLatency,
                  uint32_t writeLatency, uint64_t depth, RUWAttr ruw,
                  ArrayRef<Attribute> portNames, StringRef name,
                  NameKindEnum nameKind, ArrayRef<Attribute> annotations,
                  ArrayRef<Attribute> portAnnotations,
                  hw::InnerSymAttr innerSym) {
  result.addAttribute(
      "readLatency",
      builder.getIntegerAttr(builder.getIntegerType(32), readLatency));
  result.addAttribute(
      "writeLatency",
      builder.getIntegerAttr(builder.getIntegerType(32), writeLatency));
  result.addAttribute(
      "depth", builder.getIntegerAttr(builder.getIntegerType(64), depth));
  result.addAttribute("ruw", ::RUWAttrAttr::get(builder.getContext(), ruw));
  result.addAttribute("portNames", builder.getArrayAttr(portNames));
  result.addAttribute("name", builder.getStringAttr(name));
  result.addAttribute("nameKind",
                      NameKindEnumAttr::get(builder.getContext(), nameKind));
  result.addAttribute("annotations", builder.getArrayAttr(annotations));
  if (innerSym)
    result.addAttribute("inner_sym", innerSym);
  result.addTypes(resultTypes);

  if (portAnnotations.empty()) {
    SmallVector<Attribute, 16> portAnnotationsVec(resultTypes.size(),
                                                  builder.getArrayAttr({}));
    result.addAttribute("portAnnotations",
                        builder.getArrayAttr(portAnnotationsVec));
  } else {
    assert(portAnnotations.size() == resultTypes.size());
    result.addAttribute("portAnnotations",
                        builder.getArrayAttr(portAnnotations));
  }
}

void MemOp::build(OpBuilder &builder, OperationState &result,
                  TypeRange resultTypes, uint32_t readLatency,
                  uint32_t writeLatency, uint64_t depth, RUWAttr ruw,
                  ArrayRef<Attribute> portNames, StringRef name,
                  NameKindEnum nameKind, ArrayRef<Attribute> annotations,
                  ArrayRef<Attribute> portAnnotations, StringAttr innerSym) {
  build(builder, result, resultTypes, readLatency, writeLatency, depth, ruw,
        portNames, name, nameKind, annotations, portAnnotations,
        innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr());
}

ArrayAttr MemOp::getPortAnnotation(unsigned portIdx) {
  assert(portIdx < getNumResults() &&
         "index should be smaller than result number");
  return getPortAnnotations()[portIdx].cast<ArrayAttr>();
}

void MemOp::setAllPortAnnotations(ArrayRef<Attribute> annotations) {
  assert(annotations.size() == getNumResults() &&
         "number of annotations is not equal to result number");
  (*this)->setAttr("portAnnotations",
                   ArrayAttr::get(getContext(), annotations));
}

// Get the number of read, write and read-write ports.
void MemOp::getNumPorts(size_t &numReadPorts, size_t &numWritePorts,
                        size_t &numReadWritePorts, size_t &numDbgsPorts) {
  numReadPorts = 0;
  numWritePorts = 0;
  numReadWritePorts = 0;
  numDbgsPorts = 0;
  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    auto portKind = getPortKind(i);
    if (portKind == MemOp::PortKind::Debug)
      ++numDbgsPorts;
    else if (portKind == MemOp::PortKind::Read)
      ++numReadPorts;
    else if (portKind == MemOp::PortKind::Write) {
      ++numWritePorts;
    } else
      ++numReadWritePorts;
  }
}

/// Verify the correctness of a MemOp.
LogicalResult MemOp::verify() {

  // Store the port names as we find them. This lets us check quickly
  // for uniqueneess.
  llvm::SmallDenseSet<Attribute, 8> portNamesSet;

  // Store the previous data type. This lets us check that the data
  // type is consistent across all ports.
  FIRRTLType oldDataType;

  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    auto portName = getPortName(i);

    // Get a bundle type representing this port, stripping an outer
    // flip if it exists.  If this is not a bundle<> or
    // flip<bundle<>>, then this is an error.
    BundleType portBundleType =
        TypeSwitch<FIRRTLType, BundleType>(
            getResult(i).getType().cast<FIRRTLType>())
            .Case<BundleType>([](BundleType a) { return a; })
            .Default([](auto) { return nullptr; });

    // Require that all port names are unique.
    if (!portNamesSet.insert(portName).second) {
      emitOpError() << "has non-unique port name " << portName;
      return failure();
    }

    // Determine the kind of the memory.  If the kind cannot be
    // determined, then it's indicative of the wrong number of fields
    // in the type (but we don't know any more just yet).

    auto elt = getPortNamed(portName);
    if (!elt) {
      emitOpError() << "could not get port with name " << portName;
      return failure();
    }
    auto firrtlType = elt.getType().cast<FIRRTLType>();
    MemOp::PortKind portKind = getMemPortKindFromType(firrtlType);

    if (portKind == MemOp::PortKind::Debug &&
        !getResult(i).getType().isa<RefType>())
      return emitOpError() << "has an invalid type on port " << portName
                           << " (expected Read/Write/ReadWrite/Debug)";
    if (firrtlType.isa<RefType>() && e == 1)
      return emitOpError()
             << "cannot have only one port of debug type. Debug port can only "
                "exist alongside other read/write/read-write port";

    // Safely search for the "data" field, erroring if it can't be
    // found.
    FIRRTLBaseType dataType;
    if (portKind == MemOp::PortKind::Debug) {
      auto resType = getResult(i).getType().dyn_cast<RefType>();
      if (!(resType && resType.getType().isa<FVectorType>()))
        return emitOpError() << "debug ports must be a RefType of FVectorType";
      dataType = resType.getType().cast<FVectorType>().getElementType();
    } else {
      auto dataTypeOption = portBundleType.getElement("data");
      if (!dataTypeOption && portKind == MemOp::PortKind::ReadWrite)
        dataTypeOption = portBundleType.getElement("wdata");
      if (!dataTypeOption) {
        emitOpError() << "has no data field on port " << portName
                      << " (expected to see \"data\" for a read or write "
                         "port or \"rdata\" for a read/write port)";
        return failure();
      }
      dataType = dataTypeOption->type;
      // Read data is expected to ba a flip.
      if (portKind == MemOp::PortKind::Read) {
        // FIXME error on missing bundle flip
      }
    }

    // Error if the data type isn't passive.
    if (!dataType.isPassive()) {
      emitOpError() << "has non-passive data type on port " << portName
                    << " (memory types must be passive)";
      return failure();
    }

    // Error if the data type contains analog types.
    if (dataType.containsAnalog()) {
      emitOpError() << "has a data type that contains an analog type on port "
                    << portName
                    << " (memory types cannot contain analog types)";
      return failure();
    }

    // Check that the port type matches the kind that we determined
    // for this port.  This catches situations of extraneous port
    // fields beind included or the fields being named incorrectly.
    FIRRTLType expectedType =
        getTypeForPort(getDepth(), dataType, portKind,
                       dataType.isGround() ? getMaskBits() : 0);
    // Compute the original port type as portBundleType may have
    // stripped outer flip information.
    auto originalType = getResult(i).getType();
    if (originalType != expectedType) {
      StringRef portKindName;
      switch (portKind) {
      case MemOp::PortKind::Read:
        portKindName = "read";
        break;
      case MemOp::PortKind::Write:
        portKindName = "write";
        break;
      case MemOp::PortKind::ReadWrite:
        portKindName = "readwrite";
        break;
      case MemOp::PortKind::Debug:
        portKindName = "dbg";
        break;
      }
      emitOpError() << "has an invalid type for port " << portName
                    << " of determined kind \"" << portKindName
                    << "\" (expected " << expectedType << ", but got "
                    << originalType << ")";
      return failure();
    }

    // Error if the type of the current port was not the same as the
    // last port, but skip checking the first port.
    if (oldDataType && oldDataType != dataType) {
      emitOpError() << "port " << getPortName(i)
                    << " has a different type than port " << getPortName(i - 1)
                    << " (expected " << oldDataType << ", but got " << dataType
                    << ")";
      return failure();
    }

    oldDataType = dataType;
  }

  auto maskWidth = getMaskBits();

  auto dataWidth = getDataType().getBitWidthOrSentinel();
  if (dataWidth > 0 && maskWidth > (size_t)dataWidth)
    return emitOpError("the mask width cannot be greater than "
                       "data width");

  if (getPortAnnotations().size() != getNumResults())
    return emitOpError("the number of result annotations should be "
                       "equal to the number of results");

  return success();
}

FIRRTLType MemOp::getTypeForPort(uint64_t depth, FIRRTLBaseType dataType,
                                 PortKind portKind, size_t maskBits) {

  auto *context = dataType.getContext();
  if (portKind == PortKind::Debug)
    return RefType::get(FVectorType::get(dataType, depth));
  FIRRTLBaseType maskType;
  // maskBits not specified (==0), then get the mask type from the dataType.
  if (maskBits == 0)
    maskType = dataType.getMaskType();
  else
    maskType = UIntType::get(context, maskBits);

  auto getId = [&](StringRef name) -> StringAttr {
    return StringAttr::get(context, name);
  };

  SmallVector<BundleType::BundleElement, 7> portFields;

  auto addressType =
      UIntType::get(context, std::max(1U, llvm::Log2_64_Ceil(depth)));

  portFields.push_back({getId("addr"), false, addressType});
  portFields.push_back({getId("en"), false, UIntType::get(context, 1)});
  portFields.push_back({getId("clk"), false, ClockType::get(context)});

  switch (portKind) {
  case PortKind::Read:
    portFields.push_back({getId("data"), true, dataType});
    break;

  case PortKind::Write:
    portFields.push_back({getId("data"), false, dataType});
    portFields.push_back({getId("mask"), false, maskType});
    break;

  case PortKind::ReadWrite:
    portFields.push_back({getId("rdata"), true, dataType});
    portFields.push_back({getId("wmode"), false, UIntType::get(context, 1)});
    portFields.push_back({getId("wdata"), false, dataType});
    portFields.push_back({getId("wmask"), false, maskType});
    break;
  default:
    llvm::report_fatal_error("memory port kind not handled");
    break;
  }

  return BundleType::get(context, portFields).cast<BundleType>();
}

/// Return the name and kind of ports supported by this memory.
SmallVector<MemOp::NamedPort> MemOp::getPorts() {
  SmallVector<MemOp::NamedPort> result;
  // Each entry in the bundle is a port.
  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    // Each port is a bundle.
    auto portType = getResult(i).getType().cast<FIRRTLType>();
    result.push_back({getPortName(i), getMemPortKindFromType(portType)});
  }
  return result;
}

/// Return the kind of the specified port.
MemOp::PortKind MemOp::getPortKind(StringRef portName) {
  return getMemPortKindFromType(
      getPortNamed(portName).getType().cast<FIRRTLType>());
}

/// Return the kind of the specified port number.
MemOp::PortKind MemOp::getPortKind(size_t resultNo) {
  return getMemPortKindFromType(
      getResult(resultNo).getType().cast<FIRRTLType>());
}

/// Return the number of bits in the mask for the memory.
size_t MemOp::getMaskBits() {

  for (auto res : getResults()) {
    if (res.getType().isa<RefType>())
      continue;
    auto firstPortType = res.getType().cast<FIRRTLBaseType>();
    if (getMemPortKindFromType(firstPortType) == PortKind::Read ||
        getMemPortKindFromType(firstPortType) == PortKind::Debug)
      continue;

    FIRRTLBaseType mType;
    for (auto t : firstPortType.getPassiveType().cast<BundleType>()) {
      if (t.name.getValue().contains("mask"))
        mType = t.type;
    }
    if (mType.dyn_cast_or_null<UIntType>())
      return mType.getBitWidthOrSentinel();
  }
  // Mask of zero bits means, either there are no write/readwrite ports or the
  // mask is of aggregate type.
  return 0;
}

/// Return the data-type field of the memory, the type of each element.
FIRRTLBaseType MemOp::getDataType() {
  assert(getNumResults() != 0 && "Mems with no read/write ports are illegal");

  if (auto refType = getResult(0).getType().dyn_cast<RefType>())
    return refType.getType().cast<FVectorType>().getElementType();
  auto firstPortType = getResult(0).getType().cast<FIRRTLBaseType>();

  StringRef dataFieldName = "data";
  if (getMemPortKindFromType(firstPortType) == PortKind::ReadWrite)
    dataFieldName = "rdata";

  return firstPortType.getPassiveType().cast<BundleType>().getElementType(
      dataFieldName);
}

StringAttr MemOp::getPortName(size_t resultNo) {
  return getPortNames()[resultNo].cast<StringAttr>();
}

FIRRTLBaseType MemOp::getPortType(size_t resultNo) {
  return getResults()[resultNo].getType().cast<FIRRTLBaseType>();
}

Value MemOp::getPortNamed(StringAttr name) {
  auto namesArray = getPortNames();
  for (size_t i = 0, e = namesArray.size(); i != e; ++i) {
    if (namesArray[i] == name) {
      assert(i < getNumResults() && " names array out of sync with results");
      return getResult(i);
    }
  }
  return Value();
}

// Extract all the relevant attributes from the MemOp and return the FirMemory.
FirMemory MemOp::getSummary() {
  auto op = *this;
  size_t numReadPorts = 0;
  size_t numWritePorts = 0;
  size_t numReadWritePorts = 0;
  llvm::SmallDenseMap<Value, unsigned> clockToLeader;
  SmallVector<int32_t> writeClockIDs;

  for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
    auto portKind = op.getPortKind(i);
    if (portKind == MemOp::PortKind::Read)
      ++numReadPorts;
    else if (portKind == MemOp::PortKind::Write) {
      for (auto *a : op.getResult(i).getUsers()) {
        auto subfield = dyn_cast<SubfieldOp>(a);
        if (!subfield || subfield.getFieldIndex() != 2)
          continue;
        auto clockPort = a->getResult(0);
        for (auto *b : clockPort.getUsers()) {
          if (auto connect = dyn_cast<FConnectLike>(b)) {
            if (connect.getDest() == clockPort) {
              auto result =
                  clockToLeader.insert({circt::firrtl::getModuleScopedDriver(
                                            connect.getSrc(), true, true, true),
                                        numWritePorts});
              if (result.second) {
                writeClockIDs.push_back(numWritePorts);
              } else {
                writeClockIDs.push_back(result.first->second);
              }
            }
          }
        }
        break;
      }
      ++numWritePorts;
    } else
      ++numReadWritePorts;
  }

  size_t width = 0;
  if (auto widthV = getBitWidth(op.getDataType()))
    width = *widthV;
  else
    op.emitError("'firrtl.mem' should have simple type and known width");
  uint32_t groupID = 0;
  if (auto gID = op.getGroupIDAttr())
    groupID = gID.getUInt();
  StringAttr modName;
  if (op->hasAttr("modName"))
    modName = op->getAttrOfType<StringAttr>("modName");
  else {
    SmallString<8> clocks;
    for (auto a : writeClockIDs)
      clocks.append(Twine((char)(a + 'a')).str());
    modName = StringAttr::get(
        op->getContext(),
        llvm::formatv(
            "FIRRTLMem_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}{11}",
            numReadPorts, numWritePorts, numReadWritePorts, (size_t)width,
            op.getDepth(), op.getReadLatency(), op.getWriteLatency(),
            op.getMaskBits(), (size_t)op.getRuw(), (unsigned)hw::WUW::PortOrder,
            groupID, clocks.empty() ? "" : "_" + clocks));
  }
  return {numReadPorts,         numWritePorts,    numReadWritePorts,
          (size_t)width,        op.getDepth(),    op.getReadLatency(),
          op.getWriteLatency(), op.getMaskBits(), (size_t)op.getRuw(),
          hw::WUW::PortOrder,   writeClockIDs,    modName,
          op.getMaskBits() > 1, groupID,          op.getLoc()};
}

void MemOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  StringRef base = getName();
  if (base.empty())
    base = "mem";

  for (size_t i = 0, e = (*this)->getNumResults(); i != e; ++i) {
    setNameFn(getResult(i), (base + "_" + getPortNameStr(i)).str());
  }
}

// Construct name of the module which will be used for the memory definition.
StringAttr FirMemory::getFirMemoryName() const { return modName; }

void NodeOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

void RegOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

LogicalResult RegResetOp::verify() {
  Value reset = getResetValue();

  FIRRTLBaseType resetType = reset.getType().cast<FIRRTLBaseType>();
  FIRRTLBaseType regType = getResult().getType().cast<FIRRTLBaseType>();

  // The type of the initialiser must be equivalent to the register type.
  if (!areTypesEquivalent(resetType, regType))
    return emitError("type mismatch between register ")
           << regType << " and reset value " << resetType;

  return success();
}

void RegResetOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

void WireOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

/// If the connect is for RefType, implement the constraint for downward only
/// references. We cannot connect :
///   1. an input reference port to the output reference port.
///   2. instance reference results to each other.
/// This means, the connect can only be used for forwarding RefType module
/// ports to Instance ports.
static LogicalResult checkRefTypeFlow(Operation *connect) {
  Value dst = connect->getOperand(0);

  if (dst.getType().isa<RefType>()) {
    // RefType supports multiple readers. That is, there can be multiple
    // RefResolveOps remotely connected to a single RefSendOp.
    // But multiple RefSendOps should not be remotely connected to a single
    // RefResolveOp.
    if (!dst.hasOneUse())
      return emitError(connect->getLoc())
             << "output reference port cannot be reused by multiple operations"
             << ", it can only capture a unique dataflow";
  }
  return success();
}

/// Check if the source and sink are of appropriate flow.
static LogicalResult checkConnectFlow(Operation *connect) {
  Value dst = connect->getOperand(0);
  Value src = connect->getOperand(1);

  // TODO: Relax this to allow reads from output ports,
  // instance/memory input ports.
  if (foldFlow(src) == Flow::Sink) {
    // A sink that is a port output or instance input used as a source is okay.
    auto kind = getDeclarationKind(src);
    if (kind != DeclKind::Port && kind != DeclKind::Instance) {
      auto srcRef = getFieldRefFromValue(src);
      auto [srcName, rootKnown] = getFieldName(srcRef);
      auto diag = emitError(connect->getLoc());
      diag << "connect has invalid flow: the source expression ";
      if (rootKnown)
        diag << "\"" << srcName << "\" ";
      diag << "has sink flow, expected source or duplex flow";
      return diag.attachNote(srcRef.getLoc()) << "the source was defined here";
    }
  }
  if (foldFlow(dst) == Flow::Source) {
    auto dstRef = getFieldRefFromValue(dst);
    auto [dstName, rootKnown] = getFieldName(dstRef);
    auto diag = emitError(connect->getLoc());
    diag << "connect has invalid flow: the destination expression ";
    if (rootKnown)
      diag << "\"" << dstName << "\" ";
    diag << "has source flow, expected sink or duplex flow";
    return diag.attachNote(dstRef.getLoc())
           << "the destination was defined here";
  }
  return success();
}

LogicalResult ConnectOp::verify() {
  auto dstType = getDest().getType().cast<FIRRTLType>();
  auto srcType = getSrc().getType().cast<FIRRTLType>();
  auto dstBaseType = dstType.dyn_cast<FIRRTLBaseType>();
  auto srcBaseType = srcType.dyn_cast<FIRRTLBaseType>();
  if (!dstBaseType || !srcBaseType) {
    if (dstType != srcType)
      return emitError("may not connect different non-base types");
  } else {
    // Analog types cannot be connected and must be attached.
    if (dstBaseType.containsAnalog() || srcBaseType.containsAnalog())
      return emitError("analog types may not be connected");

    // Destination and source types must be equivalent.
    if (!areTypesEquivalent(dstBaseType, srcBaseType))
      return emitError("type mismatch between destination ")
             << dstBaseType << " and source " << srcBaseType;

    // Truncation is banned in a connection: destination bit width must be
    // greater than or equal to source bit width.
    if (!isTypeLarger(dstBaseType, srcBaseType))
      return emitError("destination ")
             << dstBaseType << " is not as wide as the source " << srcBaseType;
  }

  // Check that the flows make sense.
  if (failed(checkConnectFlow(*this)))
    return failure();

  // Check constraints on RefType.
  if (failed(checkRefTypeFlow(*this)))
    return failure();

  return success();
}

LogicalResult StrictConnectOp::verify() {
  if (auto type = getDest().getType().dyn_cast<FIRRTLType>()) {
    auto baseType = type.dyn_cast<FIRRTLBaseType>();

    // Analog types cannot be connected and must be attached.
    if (baseType && baseType.containsAnalog())
      return emitError("analog types may not be connected");
  }

  // Check that the flows make sense.
  if (failed(checkConnectFlow(*this)))
    return failure();

  // Check constraints on RefType.
  if (failed(checkRefTypeFlow(*this)))
    return failure();

  return success();
}

void WhenOp::createElseRegion() {
  assert(!hasElseRegion() && "already has an else region");
  getElseRegion().push_back(new Block());
}

void WhenOp::build(OpBuilder &builder, OperationState &result, Value condition,
                   bool withElseRegion, std::function<void()> thenCtor,
                   std::function<void()> elseCtor) {
  OpBuilder::InsertionGuard guard(builder);
  result.addOperands(condition);

  // Create "then" region.
  builder.createBlock(result.addRegion());
  if (thenCtor)
    thenCtor();

  // Create "else" region.
  Region *elseRegion = result.addRegion();
  if (withElseRegion) {
    builder.createBlock(elseRegion);
    if (elseCtor)
      elseCtor();
  }
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

/// Type inference adaptor that narrows from the very generic MLIR
/// `InferTypeOpInterface` to what we need in the FIRRTL dialect: just operands
/// and attributes, no context or regions. Also, we only ever produce a single
/// result value, so the FIRRTL-specific type inference ops directly return the
/// inferred type rather than pushing into the `results` vector.
LogicalResult impl::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, RegionRange regions, SmallVectorImpl<Type> &results,
    llvm::function_ref<FIRRTLType(ValueRange, ArrayRef<NamedAttribute>,
                                  std::optional<Location>)>
        callback) {
  auto type = callback(
      operands, attrs ? attrs.getValue() : ArrayRef<NamedAttribute>{}, loc);
  if (type) {
    results.push_back(type);
    return success();
  }
  return failure();
}

/// Get an attribute by name from a list of named attributes. Aborts if the
/// attribute does not exist.
static Attribute getAttr(ArrayRef<NamedAttribute> attrs, StringRef name) {
  for (auto attr : attrs)
    if (attr.getName() == name)
      return attr.getValue();
  llvm::report_fatal_error("attribute '" + name + "' not found");
}

/// Same as above, but casts the attribute to a specific type.
template <typename AttrClass>
AttrClass getAttr(ArrayRef<NamedAttribute> attrs, StringRef name) {
  return getAttr(attrs, name).cast<AttrClass>();
}

/// Return true if the specified operation is a firrtl expression.
bool firrtl::isExpression(Operation *op) {
  struct IsExprClassifier : public ExprVisitor<IsExprClassifier, bool> {
    bool visitInvalidExpr(Operation *op) { return false; }
    bool visitUnhandledExpr(Operation *op) { return true; }
  };

  return IsExprClassifier().dispatchExprVisitor(op);
}

void InvalidValueOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Set invalid values to have a distinct name.
  std::string name;
  if (auto ty = getType().dyn_cast<IntType>()) {
    const char *base = ty.isSigned() ? "invalid_si" : "invalid_ui";
    auto width = ty.getWidthOrSentinel();
    if (width == -1)
      name = base;
    else
      name = (Twine(base) + Twine(width)).str();
  } else if (auto ty = getType().dyn_cast<AnalogType>()) {
    auto width = ty.getWidthOrSentinel();
    if (width == -1)
      name = "invalid_analog";
    else
      name = ("invalid_analog" + Twine(width)).str();
  } else if (getType().isa<AsyncResetType>())
    name = "invalid_asyncreset";
  else if (getType().isa<ResetType>())
    name = "invalid_reset";
  else if (getType().isa<ClockType>())
    name = "invalid_clock";
  else
    name = "invalid";

  setNameFn(getResult(), name);
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValueAttr());
  p << " : ";
  p.printType(getType());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the constant value, without knowing its width.
  APInt value;
  auto loc = parser.getCurrentLocation();
  auto valueResult = parser.parseOptionalInteger(value);
  if (!valueResult.has_value())
    return parser.emitError(loc, "expected integer value");

  // Parse the result firrtl integer type.
  IntType resultType;
  if (failed(*valueResult) || parser.parseColonType(resultType) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addTypes(resultType);

  // Now that we know the width and sign of the result type, we can munge the
  // APInt as appropriate.
  if (resultType.hasWidth()) {
    auto width = (unsigned)resultType.getWidthOrSentinel();
    if (width > value.getBitWidth()) {
      // sext is always safe here, even for unsigned values, because the
      // parseOptionalInteger method will return something with a zero in the
      // top bits if it is a positive number.
      value = value.sext(width);
    } else if (width < value.getBitWidth()) {
      // The parser can return an unnecessarily wide result with leading zeros.
      // This isn't a problem, but truncating off bits is bad.
      if (value.getNumSignBits() < value.getBitWidth() - width)
        return parser.emitError(loc, "constant too large for result type ")
               << resultType;
      value = value.trunc(width);
    }
  }

  auto intType = parser.getBuilder().getIntegerType(value.getBitWidth(),
                                                    resultType.isSigned());
  auto valueAttr = parser.getBuilder().getIntegerAttr(intType, value);
  result.addAttribute("value", valueAttr);
  return success();
}

LogicalResult ConstantOp::verify() {
  // If the result type has a bitwidth, then the attribute must match its width.
  auto intType = getType().cast<IntType>();
  auto width = intType.getWidthOrSentinel();
  if (width != -1 && (int)getValue().getBitWidth() != width)
    return emitError(
        "firrtl.constant attribute bitwidth doesn't match return type");

  // The sign of the attribute's integer type must match our integer type sign.
  auto attrType = getValueAttr().getType().cast<IntegerType>();
  if (attrType.isSignless() || attrType.isSigned() != getType().isSigned())
    return emitError("firrtl.constant attribute has wrong sign");

  return success();
}

/// Build a ConstantOp from an APInt and a FIRRTL type, handling the attribute
/// formation for the 'value' attribute.
void ConstantOp::build(OpBuilder &builder, OperationState &result, IntType type,
                       const APInt &value) {
  int32_t width = type.getWidthOrSentinel();
  (void)width;
  assert((width == -1 || (int32_t)value.getBitWidth() == width) &&
         "incorrect attribute bitwidth for firrtl.constant");

  auto attr =
      IntegerAttr::get(type.getContext(), APSInt(value, !type.isSigned()));
  return build(builder, result, type, attr);
}

/// Build a ConstantOp from an APSInt, handling the attribute formation for the
/// 'value' attribute and inferring the FIRRTL type.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       const APSInt &value) {
  auto attr = IntegerAttr::get(builder.getContext(), value);
  auto type =
      IntType::get(builder.getContext(), value.isSigned(), value.getBitWidth());
  return build(builder, result, type, attr);
}

void ConstantOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // For constants in particular, propagate the value into the result name to
  // make it easier to read the IR.
  auto intTy = getType().dyn_cast<IntType>();

  // Otherwise, build a complex name with the value and type.
  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c';
  if (intTy) {
    getValue().print(specialName, /*isSigned:*/ intTy.isSigned());

    specialName << (intTy.isSigned() ? "_si" : "_ui");
    auto width = intTy.getWidthOrSentinel();
    if (width != -1)
      specialName << width;
  } else {
    getValue().print(specialName, /*isSigned:*/ false);
  }
  setNameFn(getResult(), specialName.str());
}

void SpecialConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  // SpecialConstant uses a BoolAttr, and we want to print `true` as `1`.
  p << static_cast<unsigned>(getValue());
  p << " : ";
  p.printType(getType());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

ParseResult SpecialConstantOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  // Parse the constant value.  SpecialConstant uses bool attributes, but it
  // prints as an integer.
  APInt value;
  auto loc = parser.getCurrentLocation();
  auto valueResult = parser.parseOptionalInteger(value);
  if (!valueResult.has_value())
    return parser.emitError(loc, "expected integer value");

  // Clocks and resets can only be 0 or 1.
  if (value != 0 && value != 1)
    return parser.emitError(loc, "special constants can only be 0 or 1.");

  // Parse the result firrtl type.
  Type resultType;
  if (failed(*valueResult) || parser.parseColonType(resultType) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addTypes(resultType);

  // Create the attribute.
  auto valueAttr = parser.getBuilder().getBoolAttr(value == 1);
  result.addAttribute("value", valueAttr);
  return success();
}

void SpecialConstantOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c';
  specialName << static_cast<unsigned>(getValue());
  auto type = getType();
  if (type.isa<ClockType>()) {
    specialName << "_clock";
  } else if (type.isa<ResetType>()) {
    specialName << "_reset";
  } else if (type.isa<AsyncResetType>()) {
    specialName << "_asyncreset";
  }
  setNameFn(getResult(), specialName.str());
}

// Checks that an array attr representing an aggregate constant has the correct
// shape.  This recurses on the type.
static bool checkAggConstant(Operation *op, Attribute attr,
                             FIRRTLBaseType type) {
  if (type.isGround()) {
    if (!attr.isa<IntegerAttr>()) {
      op->emitOpError("Ground type is not an integer attribute");
      return false;
    }
    return true;
  }
  auto attrlist = attr.dyn_cast<ArrayAttr>();
  if (!attrlist) {
    op->emitOpError("expected array attribute for aggregate constant");
    return false;
  }
  if (auto array = type.dyn_cast<FVectorType>()) {
    if (array.getNumElements() != attrlist.size()) {
      op->emitOpError("array attribute (")
          << attrlist.size() << ") has wrong size for vector constant ("
          << array.getNumElements() << ")";
      return false;
    }
    return llvm::all_of(attrlist, [&array, op](Attribute attr) {
      return checkAggConstant(op, attr, array.getElementType());
    });
  }
  if (auto bundle = type.dyn_cast<BundleType>()) {
    if (bundle.getNumElements() != attrlist.size()) {
      op->emitOpError("array attribute (")
          << attrlist.size() << ") has wrong size for bundle constant ("
          << bundle.getNumElements() << ")";
      return false;
    }
    for (size_t i = 0; i < bundle.getNumElements(); ++i) {
      if (bundle.getElement(i).isFlip) {
        op->emitOpError("Cannot have constant bundle type with flip");
        return false;
      }
      if (!checkAggConstant(op, attrlist[i], bundle.getElement(i).type))
        return false;
    }
    return true;
  }
  op->emitOpError("Unknown aggregate type");
  return false;
}

LogicalResult AggregateConstantOp::verify() {
  if (checkAggConstant(getOperation(), getFields(), getType()))
    return success();
  return failure();
}

LogicalResult BundleCreateOp::verify() {
  if (getType().getNumElements() != getFields().size())
    return emitOpError("number of fields doesn't match type");
  for (size_t i = 0; i < getType().getNumElements(); ++i)
    if (getType().getElement(i).type != getOperand(i).getType())
      return emitOpError("type of element doesn't match bundle for field ")
             << getType().getElement(i).name;
  // TODO: check flow
  return success();
}

LogicalResult VectorCreateOp::verify() {
  if (getType().getNumElements() != getFields().size())
    return emitOpError("number of fields doesn't match type");
  auto elemTy = getType().getElementType();
  for (size_t i = 0; i < getType().getNumElements(); ++i)
    if (elemTy != getOperand(i).getType())
      return emitOpError("type of element doesn't match vector element");
  // TODO: check flow
  return success();
}

ParseResult SubfieldOp::parse(OpAsmParser &parser, OperationState &result) {
  auto *context = parser.getContext();

  OpAsmParser::UnresolvedOperand input;
  std::string fieldName;
  Type inputType;
  if (parser.parseOperand(input) || parser.parseLSquare() ||
      parser.parseKeywordOrString(&fieldName) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(inputType))
    return failure();

  if (parser.resolveOperand(input, inputType, result.operands))
    return failure();

  auto bundleType = inputType.dyn_cast<BundleType>();
  if (!bundleType)
    return parser.emitError(parser.getNameLoc(),
                            "input must be bundle type, got ")
           << inputType;
  auto fieldIndex = bundleType.getElementIndex(fieldName);
  if (!fieldIndex)
    return parser.emitError(parser.getNameLoc(),
                            "unknown field " + fieldName + " in bundle type ")
           << bundleType;

  result.addAttribute(
      "fieldIndex",
      IntegerAttr::get(IntegerType::get(context, 32), *fieldIndex));

  SmallVector<Type> inferredReturnTypes;
  if (failed(SubfieldOp::inferReturnTypes(
          context, result.location, result.operands,
          result.attributes.getDictionary(context), result.regions,
          inferredReturnTypes)))
    return failure();
  result.addTypes(inferredReturnTypes);

  return success();
}

void SubfieldOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ' << getInput() << '[';
  printer.printKeywordOrString(getFieldName());
  printer << ']';
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("fieldIndex");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << " : " << getInput().getType();
}

LogicalResult SubfieldOp::verify() {
  if (getFieldIndex() >=
      getInput().getType().cast<BundleType>().getNumElements())
    return emitOpError("subfield element index is greater than the number "
                       "of fields in the bundle type");
  return success();
}

/// Return true if the specified operation has a constant value. This trivially
/// checks for `firrtl.constant` and friends, but also looks through subaccesses
/// and correctly handles wires driven with only constant values.
bool firrtl::isConstant(Operation *op) {
  // Worklist of ops that need to be examined that should all be constant in
  // order for the input operation to be constant.
  SmallVector<Operation *, 8> worklist({op});

  // Mutable state indicating if this op is a constant.  Assume it is a constant
  // and look for counterexamples.
  bool constant = true;

  // While we haven't found a counterexample and there are still ops in the
  // worklist, pull ops off the worklist.  If it provides a counterexample, set
  // the `constant` to false (and exit on the next loop iteration).  Otherwise,
  // look through the op or spawn off more ops to look at.
  while (constant && !(worklist.empty()))
    TypeSwitch<Operation *>(worklist.pop_back_val())
        .Case<NodeOp, AsSIntPrimOp, AsUIntPrimOp>([&](auto op) {
          if (auto definingOp = op.getInput().getDefiningOp())
            worklist.push_back(definingOp);
          constant = false;
        })
        .Case<WireOp, SubindexOp, SubfieldOp>([&](auto op) {
          for (auto &use : op.getResult().getUses())
            worklist.push_back(use.getOwner());
        })
        .Case<ConstantOp, SpecialConstantOp, AggregateConstantOp>([](auto) {})
        .Default([&](auto) { constant = false; });

  return constant;
}

/// Return true if the specified value is a constant. This trivially checks for
/// `firrtl.constant` and friends, but also looks through subaccesses and
/// correctly handles wires driven with only constant values.
bool firrtl::isConstant(Value value) {
  if (auto *op = value.getDefiningOp())
    return isConstant(op);
  return false;
}

FIRRTLType SubfieldOp::inferReturnType(ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       std::optional<Location> loc) {
  auto inType = operands[0].getType().cast<BundleType>();
  auto fieldIndex =
      getAttr<IntegerAttr>(attrs, "fieldIndex").getValue().getZExtValue();

  if (fieldIndex >= inType.getNumElements()) {
    if (loc)
      mlir::emitError(*loc, "subfield element index is greater than the number "
                            "of fields in the bundle type");
    return {};
  }

  // SubfieldOp verifier checks that the field index is valid with number of
  // subelements.
  return inType.getElement(fieldIndex).type;
}

bool SubfieldOp::isFieldFlipped() {
  auto bundle = getInput().getType().cast<BundleType>();
  return bundle.getElement(getFieldIndex()).isFlip;
}

FIRRTLType SubindexOp::inferReturnType(ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       std::optional<Location> loc) {
  auto inType = operands[0].getType();
  auto fieldIdx =
      getAttr<IntegerAttr>(attrs, "index").getValue().getZExtValue();

  if (auto vectorType = inType.dyn_cast<FVectorType>()) {
    if (fieldIdx < vectorType.getNumElements())
      return vectorType.getElementType();
    if (loc)
      mlir::emitError(*loc, "out of range index '")
          << fieldIdx << "' in vector type " << inType;
    return {};
  }

  if (loc)
    mlir::emitError(*loc, "subindex requires vector operand");
  return {};
}

FIRRTLType SubaccessOp::inferReturnType(ValueRange operands,
                                        ArrayRef<NamedAttribute> attrs,
                                        std::optional<Location> loc) {
  auto inType = operands[0].getType();
  auto indexType = operands[1].getType();

  if (!indexType.isa<UIntType>()) {
    if (loc)
      mlir::emitError(*loc, "subaccess index must be UInt type, not ")
          << indexType;
    return {};
  }

  if (auto vectorType = inType.dyn_cast<FVectorType>())
    return vectorType.getElementType();

  if (loc)
    mlir::emitError(*loc, "subaccess requires vector operand, not ") << inType;
  return {};
}

ParseResult MultibitMuxOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand index;
  SmallVector<OpAsmParser::UnresolvedOperand, 16> inputs;
  Type indexType, elemType;

  if (parser.parseOperand(index) || parser.parseComma() ||
      parser.parseOperandList(inputs) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(indexType) || parser.parseComma() ||
      parser.parseType(elemType))
    return failure();

  if (parser.resolveOperand(index, indexType, result.operands))
    return failure();

  result.addTypes(elemType);

  return parser.resolveOperands(inputs, elemType, result.operands);
}

void MultibitMuxOp::print(OpAsmPrinter &p) {
  p << " " << getIndex() << ", ";
  p.printOperands(getInputs());
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getIndex().getType() << ", " << getType();
}

FIRRTLType MultibitMuxOp::inferReturnType(ValueRange operands,
                                          ArrayRef<NamedAttribute> attrs,
                                          std::optional<Location> loc) {
  if (operands.size() < 2) {
    if (loc)
      mlir::emitError(*loc, "at least one input is required");
    return FIRRTLType();
  }

  // Check all mux inputs have the same type.
  if (!llvm::all_of(operands.drop_front(2), [&](auto op) {
        return operands[1].getType() == op.getType();
      })) {
    if (loc)
      mlir::emitError(*loc, "all inputs must have the same type");
    return FIRRTLType();
  }

  return operands[1].getType().cast<FIRRTLType>();
}

//===----------------------------------------------------------------------===//
// Binary Primitives
//===----------------------------------------------------------------------===//

/// If LHS and RHS are both UInt or SInt types, the return true and fill in the
/// width of them if known.  If unknown, return -1 for the widths.
///
/// On failure, this reports and error and returns false.  This function should
/// not be used if you don't want an error reported.
static bool isSameIntTypeKind(Type lhs, Type rhs, int32_t &lhsWidth,
                              int32_t &rhsWidth, std::optional<Location> loc) {
  // Must have two integer types with the same signedness.
  auto lhsi = lhs.dyn_cast<IntType>();
  auto rhsi = rhs.dyn_cast<IntType>();
  if (!lhsi || !rhsi || lhsi.isSigned() != rhsi.isSigned()) {
    if (loc) {
      if (lhsi && !rhsi)
        mlir::emitError(*loc, "second operand must be an integer type, not ")
            << rhs;
      else if (!lhsi && rhsi)
        mlir::emitError(*loc, "first operand must be an integer type, not ")
            << lhs;
      else if (!lhsi && !rhsi)
        mlir::emitError(*loc, "operands must be integer types, not ")
            << lhs << " and " << rhs;
      else
        mlir::emitError(*loc, "operand signedness must match");
    }
    return false;
  }

  lhsWidth = lhsi.getWidthOrSentinel();
  rhsWidth = rhsi.getWidthOrSentinel();
  return true;
}

LogicalResult impl::verifySameOperandsIntTypeKind(Operation *op) {
  assert(op->getNumOperands() == 2 &&
         "SameOperandsIntTypeKind on non-binary op");
  int32_t lhsWidth, rhsWidth;
  return success(isSameIntTypeKind(op->getOperand(0).getType(),
                                   op->getOperand(1).getType(), lhsWidth,
                                   rhsWidth, op->getLoc()));
}

LogicalResult impl::validateBinaryOpArguments(ValueRange operands,
                                              ArrayRef<NamedAttribute> attrs,
                                              Location loc) {
  if (operands.size() != 2 || !attrs.empty()) {
    mlir::emitError(loc, "operation requires two operands and no constants");
    return failure();
  }
  return success();
}

FIRRTLType impl::inferAddSubResult(FIRRTLType lhs, FIRRTLType rhs,
                                   std::optional<Location> loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = std::max(lhsWidth, rhsWidth) + 1;
  return IntType::get(lhs.getContext(), lhs.isa<SIntType>(), resultWidth);
}

FIRRTLType MulPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                            std::optional<Location> loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = lhsWidth + rhsWidth;

  return IntType::get(lhs.getContext(), lhs.isa<SIntType>(), resultWidth);
}

FIRRTLType DivPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                            std::optional<Location> loc) {
  int32_t lhsWidth, rhsWidth;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  // For unsigned, the width is the width of the numerator on the LHS.
  if (lhs.isa<UIntType>())
    return UIntType::get(lhs.getContext(), lhsWidth);

  // For signed, the width is the width of the numerator on the LHS, plus 1.
  int32_t resultWidth = lhsWidth != -1 ? lhsWidth + 1 : -1;
  return SIntType::get(lhs.getContext(), resultWidth);
}

FIRRTLType RemPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                            std::optional<Location> loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = std::min(lhsWidth, rhsWidth);
  return IntType::get(lhs.getContext(), lhs.isa<SIntType>(), resultWidth);
}

FIRRTLType impl::inferBitwiseResult(FIRRTLType lhs, FIRRTLType rhs,
                                    std::optional<Location> loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = std::max(lhsWidth, rhsWidth);
  return UIntType::get(lhs.getContext(), resultWidth);
}

FIRRTLType impl::inferComparisonResult(FIRRTLType lhs, FIRRTLType rhs,
                                       std::optional<Location> loc) {
  return UIntType::get(lhs.getContext(), 1);
}

FIRRTLType CatPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                            std::optional<Location> loc) {
  int32_t lhsWidth, rhsWidth, resultWidth = -1;
  if (!isSameIntTypeKind(lhs, rhs, lhsWidth, rhsWidth, loc))
    return {};

  if (lhsWidth != -1 && rhsWidth != -1)
    resultWidth = lhsWidth + rhsWidth;
  return UIntType::get(lhs.getContext(), resultWidth);
}

FIRRTLType DShlPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                             std::optional<Location> loc) {
  auto lhsi = lhs.dyn_cast<IntType>();
  auto rhsui = rhs.dyn_cast<UIntType>();
  if (!rhsui || !lhsi) {
    if (loc)
      mlir::emitError(*loc,
                      "first operand should be integer, second unsigned int");
    return {};
  }

  // If the left or right has unknown result type, then the operation does
  // too.
  auto width = lhsi.getWidthOrSentinel();
  if (width == -1 || !rhsui.getWidth().has_value()) {
    width = -1;
  } else {
    auto amount = *rhsui.getWidth();
    if (amount >= 32) {
      if (loc)
        mlir::emitError(*loc, "shift amount too large: second operand of dshl "
                              "is wider than 31 bits");
      return {};
    }
    int64_t newWidth = (int64_t)width + ((int64_t)1 << amount) - 1;
    if (newWidth > INT32_MAX) {
      if (loc)
        mlir::emitError(*loc, "shift amount too large: first operand shifted "
                              "by maximum amount exceeds maximum width");
      return {};
    }
    width = newWidth;
  }
  return IntType::get(lhs.getContext(), lhsi.isSigned(), width);
}

FIRRTLType DShlwPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                              std::optional<Location> loc) {
  if (!lhs.isa<IntType>() || !rhs.isa<UIntType>()) {
    if (loc)
      mlir::emitError(*loc,
                      "first operand should be integer, second unsigned int");
    return {};
  }
  return lhs;
}

FIRRTLType DShrPrimOp::inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                             std::optional<Location> loc) {
  if (!lhs.isa<IntType>() || !rhs.isa<UIntType>()) {
    if (loc)
      mlir::emitError(*loc,
                      "first operand should be integer, second unsigned int");
    return {};
  }
  return lhs;
}

//===----------------------------------------------------------------------===//
// Unary Primitives
//===----------------------------------------------------------------------===//

LogicalResult impl::validateUnaryOpArguments(ValueRange operands,
                                             ArrayRef<NamedAttribute> attrs,
                                             Location loc) {
  if (operands.size() != 1 || !attrs.empty()) {
    mlir::emitError(loc, "operation requires one operand and no constants");
    return failure();
  }
  return success();
}

FIRRTLType
SizeOfIntrinsicOp::inferUnaryReturnType(FIRRTLType arg,
                                        std::optional<Location> loc) {
  return UIntType::get(arg.getContext(), 32);
}

FIRRTLType AsSIntPrimOp::inferUnaryReturnType(FIRRTLType input,
                                              std::optional<Location> loc) {
  auto base = input.dyn_cast<FIRRTLBaseType>();
  if (!base) {
    if (loc)
      mlir::emitError(*loc, "operand must be a scalar base type");
    return {};
  }
  int32_t width = base.getBitWidthOrSentinel();
  if (width == -2) {
    if (loc)
      mlir::emitError(*loc, "operand must be a scalar type");
    return {};
  }
  return SIntType::get(input.getContext(), width);
}

FIRRTLType AsUIntPrimOp::inferUnaryReturnType(FIRRTLType input,
                                              std::optional<Location> loc) {
  auto base = input.dyn_cast<FIRRTLBaseType>();
  if (!base) {
    if (loc)
      mlir::emitError(*loc, "operand must be a scalar base type");
    return {};
  }
  int32_t width = base.getBitWidthOrSentinel();
  if (width == -2) {
    if (loc)
      mlir::emitError(*loc, "operand must be a scalar type");
    return {};
  }
  return UIntType::get(input.getContext(), width);
}

FIRRTLType
AsAsyncResetPrimOp::inferUnaryReturnType(FIRRTLType input,
                                         std::optional<Location> loc) {
  auto base = input.dyn_cast<FIRRTLBaseType>();
  if (!base) {
    if (loc)
      mlir::emitError(*loc, "operand must be single bit scalar base type");
    return {};
  }
  int32_t width = base.getBitWidthOrSentinel();
  if (width == -2 || width == 0 || width > 1) {
    if (loc)
      mlir::emitError(*loc, "operand must be single bit scalar type");
    return {};
  }
  return AsyncResetType::get(input.getContext());
}

FIRRTLType AsClockPrimOp::inferUnaryReturnType(FIRRTLType input,
                                               std::optional<Location> loc) {
  return ClockType::get(input.getContext());
}

FIRRTLType CvtPrimOp::inferUnaryReturnType(FIRRTLType input,
                                           std::optional<Location> loc) {
  if (auto uiType = input.dyn_cast<UIntType>()) {
    auto width = uiType.getWidthOrSentinel();
    if (width != -1)
      ++width;
    return SIntType::get(input.getContext(), width);
  }

  if (input.isa<SIntType>())
    return input;

  if (loc)
    mlir::emitError(*loc, "operand must have integer type");
  return {};
}

FIRRTLType NegPrimOp::inferUnaryReturnType(FIRRTLType input,
                                           std::optional<Location> loc) {
  auto inputi = input.dyn_cast<IntType>();
  if (!inputi) {
    if (loc)
      mlir::emitError(*loc, "operand must have integer type");

    return {};
  }
  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    ++width;
  return SIntType::get(input.getContext(), width);
}

FIRRTLType NotPrimOp::inferUnaryReturnType(FIRRTLType input,
                                           std::optional<Location> loc) {
  auto inputi = input.dyn_cast<IntType>();
  if (!inputi) {
    if (loc)
      mlir::emitError(*loc, "operand must have integer type");

    return {};
  }
  return UIntType::get(input.getContext(), inputi.getWidthOrSentinel());
}

FIRRTLType impl::inferReductionResult(FIRRTLType input,
                                      std::optional<Location> loc) {
  return UIntType::get(input.getContext(), 1);
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

LogicalResult BitsPrimOp::validateArguments(ValueRange operands,
                                            ArrayRef<NamedAttribute> attrs,
                                            Location loc) {
  if (operands.size() != 1 || attrs.size() != 2) {
    mlir::emitError(loc, "operation requires one operand and two constants");
    return failure();
  }
  return success();
}

FIRRTLType BitsPrimOp::inferReturnType(ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       std::optional<Location> loc) {
  auto input = operands[0].getType();
  auto high = getAttr<IntegerAttr>(attrs, "hi").getValue().getSExtValue();
  auto low = getAttr<IntegerAttr>(attrs, "lo").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (!inputi) {
    if (loc)
      mlir::emitError(*loc, "input type should be the int type but got ")
          << input;
    return {};
  }

  // High must be >= low and both most be non-negative.
  if (high < low) {
    if (loc)
      mlir::emitError(*loc,
                      "high must be equal or greater than low, but got high = ")
          << high << ", low = " << low;
    return {};
  }

  if (low < 0) {
    if (loc)
      mlir::emitError(*loc, "low must be non-negative but got ") << low;
    return {};
  }

  // If the input has staticly known width, check it.  Both and low must be
  // strictly less than width.
  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1 && high >= width) {
    if (loc)
      mlir::emitError(*loc)
          << "high must be smaller than the width of input, but got high = "
          << high << ", width = " << width;
    return {};
  }

  return UIntType::get(input.getContext(), high - low + 1);
}

LogicalResult impl::validateOneOperandOneConst(ValueRange operands,
                                               ArrayRef<NamedAttribute> attrs,
                                               Location loc) {
  if (operands.size() != 1 || attrs.size() != 1) {
    mlir::emitError(loc, "operation requires one operand and one constant");
    return failure();
  }
  return success();
}

FIRRTLType HeadPrimOp::inferReturnType(ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       std::optional<Location> loc) {
  auto input = operands[0].getType();
  auto amount = getAttr<IntegerAttr>(attrs, "amount").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    if (loc)
      mlir::emitError(*loc,
                      "operand must have integer type and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1 && amount > width) {
    if (loc)
      mlir::emitError(*loc, "amount larger than input width");
    return {};
  }

  return UIntType::get(input.getContext(), amount);
}

LogicalResult MuxPrimOp::validateArguments(ValueRange operands,
                                           ArrayRef<NamedAttribute> attrs,
                                           Location loc) {
  if (operands.size() != 3 || attrs.size() != 0) {
    mlir::emitError(loc, "operation requires three operands and no constants");
    return failure();
  }
  return success();
}

/// Infer the result type for a multiplexer given its two operand types, which
/// may be aggregates.
///
/// This essentially performs a pairwise comparison of fields and elements, as
/// follows:
/// - Identical operands inferred to their common type
/// - Integer operands inferred to the larger one if both have a known width, a
///   widthless integer otherwise.
/// - Vectors inferred based on the element type.
/// - Bundles inferred in a pairwise fashion based on the field types.
static FIRRTLBaseType inferMuxReturnType(FIRRTLBaseType high,
                                         FIRRTLBaseType low,
                                         std::optional<Location> loc) {
  // If the types are identical we're done.
  if (high == low)
    return low;

  // The base types need to be equivalent.
  if (high.getTypeID() != low.getTypeID()) {
    if (loc) {
      auto d = mlir::emitError(*loc, "incompatible mux operand types");
      d.attachNote() << "true value type:  " << high;
      d.attachNote() << "false value type: " << low;
    }
    return {};
  }

  // Two different Int types can be compatible.  If either has unknown width,
  // then return it.  If both are known but different width, then return the
  // larger one.
  if (low.isa<IntType>()) {
    int32_t highWidth = high.getBitWidthOrSentinel();
    int32_t lowWidth = low.getBitWidthOrSentinel();
    if (lowWidth == -1)
      return low;
    if (highWidth == -1)
      return high;
    return lowWidth > highWidth ? low : high;
  }

  // Infer vector types by comparing the element types.
  auto highVector = high.dyn_cast<FVectorType>();
  auto lowVector = low.dyn_cast<FVectorType>();
  if (highVector && lowVector &&
      highVector.getNumElements() == lowVector.getNumElements()) {
    auto inner = inferMuxReturnType(highVector.getElementType(),
                                    lowVector.getElementType(), loc);
    if (!inner)
      return {};
    return FVectorType::get(inner, lowVector.getNumElements());
  }

  // Infer bundle types by inferring names in a pairwise fashion.
  auto highBundle = high.dyn_cast<BundleType>();
  auto lowBundle = low.dyn_cast<BundleType>();
  if (highBundle && lowBundle) {
    auto highElements = highBundle.getElements();
    auto lowElements = lowBundle.getElements();
    size_t numElements = highElements.size();

    SmallVector<BundleType::BundleElement> newElements;
    if (numElements == lowElements.size()) {
      bool failed = false;
      for (size_t i = 0; i < numElements; ++i) {
        if (highElements[i].name != lowElements[i].name ||
            highElements[i].isFlip != lowElements[i].isFlip) {
          failed = true;
          break;
        }
        auto element = highElements[i];
        element.type =
            inferMuxReturnType(highElements[i].type, lowElements[i].type, loc);
        if (!element.type)
          return {};
        newElements.push_back(element);
      }
      if (!failed)
        return BundleType::get(low.getContext(), newElements);
    }
    if (loc) {
      auto d = mlir::emitError(*loc, "incompatible mux operand bundle fields");
      d.attachNote() << "true value type:  " << high;
      d.attachNote() << "false value type: " << low;
    }
    return {};
  }

  // If we arrive here the types of the two mux arms are fundamentally
  // incompatible.
  if (loc) {
    auto d = mlir::emitError(*loc, "invalid mux operand types");
    d.attachNote() << "true value type:  " << high;
    d.attachNote() << "false value type: " << low;
  }
  return {};
}

FIRRTLType MuxPrimOp::inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      std::optional<Location> loc) {
  auto highType = operands[1].getType().dyn_cast<FIRRTLBaseType>();
  auto lowType = operands[2].getType().dyn_cast<FIRRTLBaseType>();
  if (!highType || !lowType) {
    if (loc)
      mlir::emitError(*loc, "operands must be base type");
    return {};
  }
  return inferMuxReturnType(highType, lowType, loc);
}

FIRRTLType PadPrimOp::inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      std::optional<Location> loc) {
  auto input = operands[0].getType();
  auto amount = getAttr<IntegerAttr>(attrs, "amount").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    if (loc)
      mlir::emitError(*loc,
                      "pad input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width == -1)
    return inputi;

  width = std::max<int32_t>(width, amount);
  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType ShlPrimOp::inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      std::optional<Location> loc) {
  auto input = operands[0].getType();
  auto amount = getAttr<IntegerAttr>(attrs, "amount").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    if (loc)
      mlir::emitError(*loc,
                      "shl input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    width += amount;

  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType ShrPrimOp::inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      std::optional<Location> loc) {
  auto input = operands[0].getType();
  auto amount = getAttr<IntegerAttr>(attrs, "amount").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    if (loc)
      mlir::emitError(*loc,
                      "shr input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1)
    width = std::max<int32_t>(1, width - amount);

  return IntType::get(input.getContext(), inputi.isSigned(), width);
}

FIRRTLType TailPrimOp::inferReturnType(ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       std::optional<Location> loc) {
  auto input = operands[0].getType();
  auto amount = getAttr<IntegerAttr>(attrs, "amount").getValue().getSExtValue();

  auto inputi = input.dyn_cast<IntType>();
  if (amount < 0 || !inputi) {
    if (loc)
      mlir::emitError(*loc,
                      "tail input must be integer and amount must be >= 0");
    return {};
  }

  int32_t width = inputi.getWidthOrSentinel();
  if (width != -1) {
    if (width < amount) {
      if (loc)
        mlir::emitError(*loc,
                        "amount must be less than or equal operand width");
      return {};
    }
    width -= amount;
  }

  return IntType::get(input.getContext(), false, width);
}

//===----------------------------------------------------------------------===//
// Verif Expressions
//===----------------------------------------------------------------------===//

FIRRTLType IsXIntrinsicOp::inferReturnType(ValueRange operands,
                                           ArrayRef<NamedAttribute> attrs,
                                           std::optional<Location> loc) {
  return UIntType::get(operands[0].getContext(), 1);
}

FIRRTLType
PlusArgsTestIntrinsicOp::inferReturnType(ValueRange operands,
                                         ArrayRef<NamedAttribute> attrs,
                                         std::optional<Location> loc) {
  return UIntType::get(attrs[0].getName().getContext(), 1);
}

//===----------------------------------------------------------------------===//
// VerbatimExprOp
//===----------------------------------------------------------------------===//

void VerbatimExprOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // If the text is macro like, then use a pretty name.  We only take the
  // text up to a weird character (like a paren) and currently ignore
  // parenthesized expressions.
  auto isOkCharacter = [](char c) { return llvm::isAlnum(c) || c == '_'; };
  auto name = getText();
  // Ignore a leading ` in macro name.
  if (name.startswith("`"))
    name = name.drop_front();
  name = name.take_while(isOkCharacter);
  if (!name.empty())
    setNameFn(getResult(), name);
}

//===----------------------------------------------------------------------===//
// VerbatimWireOp
//===----------------------------------------------------------------------===//

void VerbatimWireOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // If the text is macro like, then use a pretty name.  We only take the
  // text up to a weird character (like a paren) and currently ignore
  // parenthesized expressions.
  auto isOkCharacter = [](char c) { return llvm::isAlnum(c) || c == '_'; };
  auto name = getText();
  // Ignore a leading ` in macro name.
  if (name.startswith("`"))
    name = name.drop_front();
  name = name.take_while(isOkCharacter);
  if (!name.empty())
    setNameFn(getResult(), name);
}

//===----------------------------------------------------------------------===//
// Conversions to/from structs in the standard dialect.
//===----------------------------------------------------------------------===//

LogicalResult HWStructCastOp::verify() {
  // We must have a bundle and a struct, with matching pairwise fields
  BundleType bundleType;
  hw::StructType structType;
  if ((bundleType = getOperand().getType().dyn_cast<BundleType>())) {
    structType = getType().dyn_cast<hw::StructType>();
    if (!structType)
      return emitError("result type must be a struct");
  } else if ((bundleType = getType().dyn_cast<BundleType>())) {
    structType = getOperand().getType().dyn_cast<hw::StructType>();
    if (!structType)
      return emitError("operand type must be a struct");
  } else {
    return emitError("either source or result type must be a bundle type");
  }

  auto firFields = bundleType.getElements();
  auto hwFields = structType.getElements();
  if (firFields.size() != hwFields.size())
    return emitError("bundle and struct have different number of fields");

  for (size_t findex = 0, fend = firFields.size(); findex < fend; ++findex) {
    if (firFields[findex].name.getValue() != hwFields[findex].name)
      return emitError("field names don't match '")
             << firFields[findex].name.getValue() << "', '"
             << hwFields[findex].name.getValue() << "'";
    int64_t firWidth =
        FIRRTLBaseType(firFields[findex].type).getBitWidthOrSentinel();
    int64_t hwWidth = hw::getBitWidth(hwFields[findex].type);
    if (firWidth > 0 && hwWidth > 0 && firWidth != hwWidth)
      return emitError("size of field '")
             << hwFields[findex].name.getValue() << "' don't match " << firWidth
             << ", " << hwWidth;
  }

  return success();
}

LogicalResult BitCastOp::verify() {
  auto inTypeBits = getBitWidth(getOperand().getType().cast<FIRRTLBaseType>());
  auto resTypeBits = getBitWidth(getType());
  if (inTypeBits.has_value() && resTypeBits.has_value()) {
    // Bitwidths must match for valid bit
    if (*inTypeBits == *resTypeBits)
      return success();
    return emitError("the bitwidth of input (")
           << *inTypeBits << ") and result (" << *resTypeBits
           << ") don't match";
  }
  if (!inTypeBits.has_value())
    return emitError("bitwidth cannot be determined for input operand type ")
           << getOperand().getType();
  return emitError("bitwidth cannot be determined for result type ")
         << getType();
}

//===----------------------------------------------------------------------===//
// Custom attr-dict Directive that Elides Annotations
//===----------------------------------------------------------------------===//

/// Parse an optional attribute dictionary, adding an empty 'annotations'
/// attribute if not specified.
static ParseResult parseElideAnnotations(OpAsmParser &parser,
                                         NamedAttrList &resultAttrs) {
  auto result = parser.parseOptionalAttrDict(resultAttrs);
  if (!resultAttrs.get("annotations"))
    resultAttrs.append("annotations", parser.getBuilder().getArrayAttr({}));

  return result;
}

static void printElideAnnotations(OpAsmPrinter &p, Operation *op,
                                  DictionaryAttr attr,
                                  ArrayRef<StringRef> extraElides = {}) {
  SmallVector<StringRef> elidedAttrs(extraElides.begin(), extraElides.end());
  // Elide "annotations" if it is empty.
  if (op->getAttrOfType<ArrayAttr>("annotations").empty())
    elidedAttrs.push_back("annotations");
  // Elide "nameKind".
  elidedAttrs.push_back("nameKind");

  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

/// Parse an optional attribute dictionary, adding empty 'annotations' and
/// 'portAnnotations' attributes if not specified.
static ParseResult parseElidePortAnnotations(OpAsmParser &parser,
                                             NamedAttrList &resultAttrs) {
  auto result = parseElideAnnotations(parser, resultAttrs);

  if (!resultAttrs.get("portAnnotations")) {
    SmallVector<Attribute, 16> portAnnotations(
        parser.getNumResults(), parser.getBuilder().getArrayAttr({}));
    resultAttrs.append("portAnnotations",
                       parser.getBuilder().getArrayAttr(portAnnotations));
  }
  return result;
}

// Elide 'annotations' and 'portAnnotations' attributes if they are empty.
static void printElidePortAnnotations(OpAsmPrinter &p, Operation *op,
                                      DictionaryAttr attr,
                                      ArrayRef<StringRef> extraElides = {}) {
  SmallVector<StringRef, 2> elidedAttrs(extraElides.begin(), extraElides.end());

  if (llvm::all_of(op->getAttrOfType<ArrayAttr>("portAnnotations"),
                   [&](Attribute a) { return a.cast<ArrayAttr>().empty(); }))
    elidedAttrs.push_back("portAnnotations");
  printElideAnnotations(p, op, attr, elidedAttrs);
}

//===----------------------------------------------------------------------===//
// NameKind Custom Directive
//===----------------------------------------------------------------------===//

static ParseResult parseNameKind(OpAsmParser &parser,
                                 firrtl::NameKindEnumAttr &result) {
  StringRef keyword;

  if (!parser.parseOptionalKeyword(&keyword,
                                   {"interesting_name", "droppable_name"})) {
    auto kind = symbolizeNameKindEnum(keyword);
    result = NameKindEnumAttr::get(parser.getContext(), kind.value());
    return success();
  }

  // Default is droppable name.
  result =
      NameKindEnumAttr::get(parser.getContext(), NameKindEnum::DroppableName);
  return success();
}

static void printNameKind(OpAsmPrinter &p, Operation *op,
                          firrtl::NameKindEnumAttr attr,
                          ArrayRef<StringRef> extraElides = {}) {
  if (attr.getValue() != NameKindEnum::DroppableName)
    p << stringifyNameKindEnum(attr.getValue());
}

//===----------------------------------------------------------------------===//
// ImplicitSSAName Custom Directive
//===----------------------------------------------------------------------===//

static ParseResult parseImplicitSSAName(OpAsmParser &parser,
                                        NamedAttrList &resultAttrs) {
  if (parseElideAnnotations(parser, resultAttrs))
    return failure();

  // If the attribute dictionary contains no 'name' attribute, infer it from
  // the SSA name (if specified).
  if (resultAttrs.get("name"))
    return success();

  auto resultName = parser.getResultName(0).first;
  if (!resultName.empty() && isdigit(resultName[0]))
    resultName = "";
  auto nameAttr = parser.getBuilder().getStringAttr(resultName);
  auto *context = parser.getBuilder().getContext();
  resultAttrs.push_back({StringAttr::get(context, "name"), nameAttr});
  return success();
}

static void printImplicitSSAName(OpAsmPrinter &p, Operation *op,
                                 DictionaryAttr attr,
                                 ArrayRef<StringRef> extraElides = {}) {
  // List of attributes to elide when printing the dictionary.
  SmallVector<StringRef, 2> elides(extraElides.begin(), extraElides.end());
  elides.push_back(hw::InnerName::getInnerNameAttrName());

  // Note that we only need to print the "name" attribute if the asmprinter
  // result name disagrees with it.  This can happen in strange cases, e.g.
  // when there are conflicts.
  SmallString<32> resultNameStr;
  llvm::raw_svector_ostream tmpStream(resultNameStr);
  p.printOperand(op->getResult(0), tmpStream);
  auto actualName = tmpStream.str().drop_front();
  auto expectedName = op->getAttrOfType<StringAttr>("name").getValue();
  // Anonymous names are printed as digits, which is fine.
  if (actualName == expectedName ||
      (expectedName.empty() && isdigit(actualName[0])))
    elides.push_back("name");

  printElideAnnotations(p, op, attr, elides);
}

//===----------------------------------------------------------------------===//
// MemOp Custom attr-dict Directive
//===----------------------------------------------------------------------===//

static ParseResult parseMemOp(OpAsmParser &parser, NamedAttrList &resultAttrs) {
  return parseElidePortAnnotations(parser, resultAttrs);
}

/// Always elide "ruw" and elide "annotations" if it exists or if it is empty.
static void printMemOp(OpAsmPrinter &p, Operation *op, DictionaryAttr attr) {
  // "ruw" and "inner_sym" is always elided.
  printElidePortAnnotations(p, op, attr, {"ruw", "inner_sym"});
}

//===----------------------------------------------------------------------===//
// Miscellaneous custom elision logic.
//===----------------------------------------------------------------------===//

static ParseResult parseElideEmptyName(OpAsmParser &p,
                                       NamedAttrList &resultAttrs) {
  auto result = p.parseOptionalAttrDict(resultAttrs);
  if (!resultAttrs.get("name"))
    resultAttrs.append("name", p.getBuilder().getStringAttr(""));

  return result;
}

static void printElideEmptyName(OpAsmPrinter &p, Operation *op,
                                DictionaryAttr attr,
                                ArrayRef<StringRef> extraElides = {}) {
  SmallVector<StringRef> elides(extraElides.begin(), extraElides.end());
  if (op->getAttrOfType<StringAttr>("name").getValue().empty())
    elides.push_back("name");

  p.printOptionalAttrDict(op->getAttrs(), elides);
}

static ParseResult parsePrintfAttrs(OpAsmParser &p,
                                    NamedAttrList &resultAttrs) {
  return parseElideEmptyName(p, resultAttrs);
}

static void printPrintfAttrs(OpAsmPrinter &p, Operation *op,
                             DictionaryAttr attr) {
  printElideEmptyName(p, op, attr, {"formatString"});
}

static ParseResult parseStopAttrs(OpAsmParser &p, NamedAttrList &resultAttrs) {
  return parseElideEmptyName(p, resultAttrs);
}

static void printStopAttrs(OpAsmPrinter &p, Operation *op,
                           DictionaryAttr attr) {
  printElideEmptyName(p, op, attr, {"exitCode"});
}

static ParseResult parseVerifAttrs(OpAsmParser &p, NamedAttrList &resultAttrs) {
  return parseElideEmptyName(p, resultAttrs);
}

static void printVerifAttrs(OpAsmPrinter &p, Operation *op,
                            DictionaryAttr attr) {
  printElideEmptyName(p, op, attr, {"message"});
}

//===----------------------------------------------------------------------===//
// Various namers.
//===----------------------------------------------------------------------===//

static void genericAsmResultNames(Operation *op,
                                  OpAsmSetValueNameFn setNameFn) {
  // Many firrtl dialect operations have an optional 'name' attribute.  If
  // present, use it.
  if (op->getNumResults() == 1)
    if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
      setNameFn(op->getResult(0), nameAttr.getValue());
}

void AddPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void AndPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void AndRPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void SizeOfIntrinsicOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void AsAsyncResetPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void AsClockPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void AsSIntPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void AsUIntPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void BitsPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void CatPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void CvtPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void DShlPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void DShlwPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void DShrPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void DivPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void EQPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void GEQPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void GTPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void HeadPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void IsXIntrinsicOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void PlusArgsValueIntrinsicOp::getAsmResultNames(
    OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void PlusArgsTestIntrinsicOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void LEQPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void LTPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void MulPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void MultibitMuxOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void MuxPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void NEQPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void NegPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void NotPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void OrPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void OrRPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void PadPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void RemPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void ShlPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}
void ShrPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void SubPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void SubaccessOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void SubfieldOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void SubindexOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void TailPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void XorPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void XorRPrimOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

//===----------------------------------------------------------------------===//
// RefOps
//===----------------------------------------------------------------------===//

FIRRTLType RefResolveOp::inferReturnType(ValueRange operands,
                                         ArrayRef<NamedAttribute> attrs,
                                         std::optional<Location> loc) {
  auto inType = operands[0].getType();
  auto inRefType = inType.dyn_cast<RefType>();
  if (!inRefType) {
    if (loc)
      mlir::emitError(*loc, "ref.resolve operand must be ref type, not ")
          << inType;
    return {};
  }
  return inRefType.getType();
}

FIRRTLType RefSendOp::inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      std::optional<Location> loc) {
  auto inType = operands[0].getType();
  auto inBaseType = inType.dyn_cast<FIRRTLBaseType>();
  if (!inBaseType) {
    if (loc)
      mlir::emitError(*loc, "ref.send operand must be base type, not ")
          << inType;
    return {};
  }
  return RefType::get(inBaseType);
}

void RefResolveOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void RefSendOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

void RefSubOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  genericAsmResultNames(*this, setNameFn);
}

FIRRTLType RefSubOp::inferReturnType(ValueRange operands,
                                     ArrayRef<NamedAttribute> attrs,
                                     std::optional<Location> loc) {
  auto inType = operands[0].getType().cast<RefType>().getType();
  auto fieldIdx =
      getAttr<IntegerAttr>(attrs, "index").getValue().getZExtValue();

  if (auto vectorType = inType.dyn_cast<FVectorType>()) {
    if (fieldIdx < vectorType.getNumElements())
      return RefType::get(vectorType.getElementType());
    if (loc)
      mlir::emitError(*loc, "out of range index '")
          << fieldIdx << "' in RefType of vector type "
          << operands[0].getType();
    return {};
  }
  if (auto bundleType = inType.dyn_cast<BundleType>()) {
    if (fieldIdx >= bundleType.getNumElements()) {
      if (loc)
        mlir::emitError(*loc,
                        "subfield element index is greater than the number "
                        "of fields in the bundle type");
      return {};
    }
    return RefType::get(bundleType.getElement(fieldIdx).type);
  }

  if (loc)
    mlir::emitError(
        *loc, "ref.sub op requires a RefType of vector or bundle base type");
  return {};
}

//===----------------------------------------------------------------------===//
// TblGen Generated Logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTL.cpp.inc"
