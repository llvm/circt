//===- LowerOpenAggs.cpp - Lower Open Aggregate Types -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerOpenAggs pass.  This pass replaces the open
// aggregate types with hardware aggregates, with non-hardware fields
// expanded out as with LowerTypes.
//
// This pass supports reference and property types.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/FieldRefCache.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "firrtl-lower-open-aggs"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWEROPENAGGS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {

/// Information on non-hw (ref/prop) elements.
struct NonHWField {
  /// Type of the field, not a hardware type.
  FIRRTLType type;
  /// FieldID relative to base of converted type.
  uint64_t fieldID;
  /// Relative orientation.  False means aligned.
  bool isFlip;
  /// String suffix naming this field.
  SmallString<16> suffix;

  /// Print this structure to the specified stream.
  void print(raw_ostream &os, unsigned indent = 0) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print this structure to llvm::errs().
  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }
#endif
};

/// Structure that describes how a given operation with a type (which may or may
/// not contain non-hw typess) should be lowered to one or more operations with
/// other types.
struct MappingInfo {
  /// Preserve this type.  Map any uses of old directly to new.
  bool identity;

  // When not identity, the type will be split:

  /// Type of the hardware-only portion.  May be null, indicating all non-hw.
  Type hwType;

  /// List of the individual non-hw fields to be split out.
  SmallVector<NonHWField, 0> fields;

  /// List of fieldID's of interior nodes that map to nothing.  HW-only
  /// projection is empty, and not leaf.
  SmallVector<uint64_t, 0> mapToNullInteriors;

  hw::InnerSymAttr newSym = {};

  /// Determine number of types this argument maps to.
  size_t count(bool includeErased = false) const {
    if (identity)
      return 1;
    return fields.size() + (hwType ? 1 : 0) + (includeErased ? 1 : 0);
  }

  /// Print this structure to the specified stream.
  void print(raw_ostream &os, unsigned indent = 0) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print this structure to llvm::errs().
  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }
#endif
};

} // namespace

void NonHWField::print(llvm::raw_ostream &os, unsigned indent) const {
  os << llvm::formatv("{0}- type: {2}\n"
                      "{1}fieldID: {3}\n"
                      "{1}isFlip: {4}\n"
                      "{1}suffix: \"{5}\"\n",
                      llvm::fmt_pad("", indent, 0),
                      llvm::fmt_pad("", indent + 2, 0), type, fieldID, isFlip,
                      suffix);
}
void MappingInfo::print(llvm::raw_ostream &os, unsigned indent) const {
  if (identity) {
    os << "<identity>";
    return;
  }

  os.indent(indent) << "hardware: ";
  if (hwType)
    os << hwType;
  else
    os << "<none>";
  os << "\n";

  os.indent(indent) << "non-hardware:\n";
  for (auto &field : fields)
    field.print(os, indent + 2);

  os.indent(indent) << "mappedToNull:\n";
  for (auto &null : mapToNullInteriors)
    os.indent(indent + 2) << "- " << null << "\n";

  os.indent(indent) << "newSym: ";
  if (newSym)
    os << newSym;
  else
    os << "<empty>";
}

template <typename Range>
LogicalResult walkMappings(
    Range &&range, bool includeErased,
    llvm::function_ref<LogicalResult(size_t, MappingInfo &, size_t)> callback) {
  size_t count = 0;
  for (const auto &[index, pmi] : llvm::enumerate(range)) {
    if (failed(callback(index, pmi, count)))
      return failure();
    count += pmi.count(includeErased);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Visitor
//===----------------------------------------------------------------------===//

namespace {
class Visitor : public FIRRTLVisitor<Visitor, LogicalResult> {
public:
  explicit Visitor(MLIRContext *context) : context(context){};

  /// Entrypoint.
  LogicalResult visit(FModuleLike mod);

  using FIRRTLVisitor<Visitor, LogicalResult>::visitDecl;
  using FIRRTLVisitor<Visitor, LogicalResult>::visitExpr;
  using FIRRTLVisitor<Visitor, LogicalResult>::visitStmt;

  LogicalResult visitDecl(InstanceOp op);
  LogicalResult visitDecl(WireOp op);

  LogicalResult visitExpr(OpenSubfieldOp op);
  LogicalResult visitExpr(OpenSubindexOp op);

  LogicalResult visitUnhandledOp(Operation *op) {
    auto notOpenAggType = [](auto type) {
      return !isa<OpenBundleType, OpenVectorType>(type);
    };
    if (!llvm::all_of(op->getOperandTypes(), notOpenAggType) ||
        !llvm::all_of(op->getResultTypes(), notOpenAggType))
      return op->emitOpError(
          "unhandled use or producer of types containing non-hw types");
    return success();
  }

  LogicalResult visitInvalidOp(Operation *op) { return visitUnhandledOp(op); }

  /// Whether any changes were made.
  bool madeChanges() const { return changesMade; }

private:
  /// Convert a type to its HW-only projection, adjusting symbols.  Gather
  /// non-hw elements encountered and their names / positions.  Returns a
  /// MappingInfo with its findings.
  FailureOr<MappingInfo> mapType(Type type, Location errorLoc,
                                 hw::InnerSymAttr sym = {});

  /// Helper to record changes that may have been made.
  void recordChanges(bool changed) {
    if (changed)
      changesMade = true;
  }

  MLIRContext *context;

  /// Map non-HW fields to their new Value.
  /// Null value indicates no equivalent (dead).
  /// These values are available wherever the root is used.
  DenseMap<FieldRef, Value> nonHWValues;

  /// Map from original to its hw-only aggregate equivalent.
  DenseMap<Value, Value> hwOnlyAggMap;

  /// List of operations to erase at the end.
  SmallVector<Operation *> opsToErase;

  /// FieldRef cache.  Be careful to only use this for operations
  /// in the original IR / not mutated.
  FieldRefCache refs;

  /// Whether IR was changed.
  bool changesMade = false;
};
} // namespace

LogicalResult Visitor::visit(FModuleLike mod) {
  auto ports = mod.getPorts();

  SmallVector<MappingInfo, 16> portMappings;
  for (auto &port : ports) {
    auto pmi = mapType(port.type, port.loc, port.sym);
    if (failed(pmi))
      return failure();
    portMappings.push_back(*pmi);
  }

  /// Total number of types mapped to.
  /// Include erased ports.
  size_t countWithErased = 0;
  for (auto &pmi : portMappings)
    countWithErased += pmi.count(/*includeErased=*/true);

  /// Ports to add.
  SmallVector<std::pair<unsigned, PortInfo>> newPorts;

  /// Ports to remove.
  BitVector portsToErase(countWithErased);

  /// Go through each port mapping, gathering information about all new ports.
  LLVM_DEBUG({
    llvm::dbgs().indent(2) << "- name: "
                           << cast<mlir::SymbolOpInterface>(*mod).getNameAttr()
                           << "\n";
    llvm::dbgs().indent(4) << "ports:\n";
  });
  auto result = walkMappings(
      portMappings, /*includeErased=*/true,
      [&](auto index, auto &pmi, auto newIndex) -> LogicalResult {
        LLVM_DEBUG({
          llvm::dbgs().indent(6) << "- name: " << ports[index].name << "\n";
          llvm::dbgs().indent(8) << "type: " << ports[index].type << "\n";
          llvm::dbgs().indent(8) << "mapping:\n";
          pmi.print(llvm::dbgs(), /*indent=*/10);
          llvm::dbgs() << "\n";
        });
        // Index for inserting new points next to this point.
        // (Immediately after current port's index).
        auto idxOfInsertPoint = index + 1;

        if (pmi.identity)
          return success();

        auto &port = ports[index];

        // If not identity, mark this port for eventual removal.
        portsToErase.set(newIndex);

        // Create new hw-only port, this will generally replace this port.
        if (pmi.hwType) {
          auto newPort = port;
          newPort.type = pmi.hwType;
          newPort.sym = pmi.newSym;
          newPorts.emplace_back(idxOfInsertPoint, newPort);

          assert(!port.sym ||
                 (pmi.newSym && port.sym.size() == pmi.newSym.size()));

          // If want to run this pass later, need to fixup annotations.
          if (!port.annotations.empty())
            return mlir::emitError(port.loc)
                   << "annotations on open aggregates not handled yet";
        } else {
          assert(!port.sym && !pmi.newSym);
          if (!port.annotations.empty())
            return mlir::emitError(port.loc)
                   << "annotations found on aggregate with no HW";
        }

        // Create ports for each non-hw field.
        for (const auto &[findex, field] : llvm::enumerate(pmi.fields)) {
          auto name = StringAttr::get(context,
                                      Twine(port.name.strref()) + field.suffix);
          auto orientation =
              (Direction)((unsigned)port.direction ^ field.isFlip);
          PortInfo pi(name, field.type, orientation, /*symName=*/StringAttr{},
                      port.loc, std::nullopt, port.domains);
          newPorts.emplace_back(idxOfInsertPoint, pi);
        }
        return success();
      });
  if (failed(result))
    return failure();

  // Insert the new ports!
  mod.insertPorts(newPorts);
  recordChanges(!newPorts.empty());

  assert(mod->getNumRegions() == 1);

  // (helper to determine/get the body block if present)
  auto getBodyBlock = [](auto mod) {
    auto &blocks = mod->getRegion(0).getBlocks();
    return !blocks.empty() ? &blocks.front() : nullptr;
  };

  // Process body block.
  // Create mapping for ports, then visit all operations within.
  if (auto *block = getBodyBlock(mod)) {
    // Create mappings for split ports.
    auto result =
        walkMappings(portMappings, /*includeErased=*/true,
                     [&](auto index, MappingInfo &pmi, auto newIndex) {
                       // Nothing to do for identity.
                       if (pmi.identity)
                         return success();

                       // newIndex is index of this port after insertion.
                       // This will be removed.
                       assert(portsToErase.test(newIndex));
                       auto oldPort = block->getArgument(newIndex);
                       auto newPortIndex = newIndex;

                       // Create mappings for split ports.
                       if (pmi.hwType)
                         hwOnlyAggMap[oldPort] =
                             block->getArgument(++newPortIndex);

                       for (auto &field : pmi.fields) {
                         auto ref = FieldRef(oldPort, field.fieldID);
                         auto newVal = block->getArgument(++newPortIndex);
                         nonHWValues[ref] = newVal;
                       }
                       for (auto fieldID : pmi.mapToNullInteriors) {
                         auto ref = FieldRef(oldPort, fieldID);
                         assert(!nonHWValues.count(ref));
                         nonHWValues[ref] = {};
                       }

                       return success();
                     });
    if (failed(result))
      return failure();

    // Walk the module.
    LLVM_DEBUG(llvm::dbgs().indent(4) << "body:\n");
    if (block
            ->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
              return dispatchVisitor(op);
            })
            .wasInterrupted())
      return failure();

    assert(opsToErase.empty() || madeChanges());

    // Cleanup dead operations.
    for (auto &op : llvm::reverse(opsToErase))
      op->erase();
  }

  // Drop dead ports.
  mod.erasePorts(portsToErase);
  recordChanges(portsToErase.any());

  LLVM_DEBUG(refs.printStats(llvm::dbgs()));

  return success();
}

LogicalResult Visitor::visitExpr(OpenSubfieldOp op) {
  // Changes will be made.
  recordChanges(true);

  // We're indexing into an OpenBundle, which contains some non-hw elements and
  // may contain hw elements.

  // By the time this is reached, the "root" storage for the input
  // has already been handled and mapped to its new location(s),
  // such that the hardware-only contents are split from non-hw.

  // If there is a hardware portion selected by this operation,
  // create a "closed" subfieldop using the hardware-only new storage,
  // and add an entry mapping our old (soon, dead) result to
  // this new hw-only result (of the subfieldop).

  // Downstream indexing operations will expect that they can
  // still chase up through this operation, and that they will find
  // the hw-only portion in the map.

  // If this operation selects a non-hw element (not mixed),
  // look up where that ref now lives and update all users to use that instead.
  // (This case falls under "this selects only non-hw", which means
  // that this operation is now dead).

  // In all cases, this operation will be dead and should be removed.
  opsToErase.push_back(op);

  // Chase this to its original root.
  // If the FieldRef for this selection has a new home,
  // RAUW to that value and this op is dead.
  auto resultRef = refs.getFieldRefFromValue(op.getResult());
  auto nonHWForResult = nonHWValues.find(resultRef);
  if (nonHWForResult != nonHWValues.end()) {
    // If has nonHW portion, RAUW to it.
    if (auto newResult = nonHWForResult->second) {
      assert(op.getResult().getType() == newResult.getType());
      assert(!type_isa<FIRRTLBaseType>(newResult.getType()));
      op.getResult().replaceAllUsesWith(newResult);
    }
    return success();
  }

  assert(hwOnlyAggMap.count(op.getInput()));

  auto newInput = hwOnlyAggMap[op.getInput()];
  assert(newInput);

  auto bundleType = type_cast<BundleType>(newInput.getType());

  // Recompute the "actual" index for this field, it may have changed.
  auto fieldName = op.getFieldName();
  auto newFieldIndex = bundleType.getElementIndex(fieldName);
  assert(newFieldIndex.has_value());

  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto newOp = SubfieldOp::create(builder, newInput, *newFieldIndex);
  if (auto name = op->getAttrOfType<StringAttr>("name"))
    newOp->setAttr("name", name);

  hwOnlyAggMap[op.getResult()] = newOp;

  if (type_isa<FIRRTLBaseType>(op.getType()))
    op.getResult().replaceAllUsesWith(newOp.getResult());

  return success();
}

LogicalResult Visitor::visitExpr(OpenSubindexOp op) {
  // Changes will be made.
  recordChanges(true);

  // In all cases, this operation will be dead and should be removed.
  opsToErase.push_back(op);

  // Chase this to its original root.
  // If the FieldRef for this selection has a new home,
  // RAUW to that value and this op is dead.
  auto resultRef = refs.getFieldRefFromValue(op.getResult());
  auto nonHWForResult = nonHWValues.find(resultRef);
  if (nonHWForResult != nonHWValues.end()) {
    // If has nonHW portion, RAUW to it.
    if (auto newResult = nonHWForResult->second) {
      assert(op.getResult().getType() == newResult.getType());
      assert(!type_isa<FIRRTLBaseType>(newResult.getType()));
      op.getResult().replaceAllUsesWith(newResult);
    }
    return success();
  }

  assert(hwOnlyAggMap.count(op.getInput()));

  auto newInput = hwOnlyAggMap[op.getInput()];
  assert(newInput);

  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto newOp = SubindexOp::create(builder, newInput, op.getIndex());
  if (auto name = op->getAttrOfType<StringAttr>("name"))
    newOp->setAttr("name", name);

  hwOnlyAggMap[op.getResult()] = newOp;

  if (type_isa<FIRRTLBaseType>(op.getType()))
    op.getResult().replaceAllUsesWith(newOp.getResult());
  return success();
}

LogicalResult Visitor::visitDecl(InstanceOp op) {
  // Rewrite ports same strategy as for modules.

  SmallVector<MappingInfo, 16> portMappings;

  for (auto type : op.getResultTypes()) {
    auto pmi = mapType(type, op.getLoc());
    if (failed(pmi))
      return failure();
    portMappings.push_back(*pmi);
  }

  /// Total number of types mapped to.
  size_t countWithErased = 0;
  for (auto &pmi : portMappings)
    countWithErased += pmi.count(/*includeErased=*/true);

  /// Ports to add.
  SmallVector<std::pair<unsigned, PortInfo>> newPorts;

  /// Ports to remove.
  BitVector portsToErase(countWithErased);

  /// Go through each port mapping, gathering information about all new ports.
  LLVM_DEBUG({
    llvm::dbgs().indent(6) << "- instance:\n";
    llvm::dbgs().indent(10) << "name: " << op.getInstanceNameAttr() << "\n";
    llvm::dbgs().indent(10) << "module: " << op.getModuleNameAttr() << "\n";
    llvm::dbgs().indent(10) << "ports:\n";
  });
  auto result = walkMappings(
      portMappings, /*includeErased=*/true,
      [&](auto index, auto &pmi, auto newIndex) -> LogicalResult {
        LLVM_DEBUG({
          llvm::dbgs().indent(12)
              << "- name: " << op.getPortName(index) << "\n";
          llvm::dbgs().indent(14) << "type: " << op.getType(index) << "\n";
          llvm::dbgs().indent(14) << "mapping:\n";
          pmi.print(llvm::dbgs(), /*indent=*/16);
          llvm::dbgs() << "\n";
        });
        // Index for inserting new points next to this point.
        // (Immediately after current port's index).
        auto idxOfInsertPoint = index + 1;

        if (pmi.identity)
          return success();

        // If not identity, mark this port for eventual removal.
        portsToErase.set(newIndex);

        auto portName = op.getPortNameAttr(index);
        auto portDirection = op.getPortDirection(index);
        auto portDomain = op.getPortDomain(index);
        auto loc = op.getLoc();

        // Create new hw-only port, this will generally replace this port.
        if (pmi.hwType) {
          PortInfo hwPort(portName, pmi.hwType, portDirection,
                          /*symName=*/StringAttr{}, loc,
                          AnnotationSet(op.getPortAnnotation(index)),
                          portDomain);
          newPorts.emplace_back(idxOfInsertPoint, hwPort);

          // If want to run this pass later, need to fixup annotations.
          if (!op.getPortAnnotation(index).empty())
            return mlir::emitError(op.getLoc())
                   << "annotations on open aggregates not handled yet";
        } else {
          if (!op.getPortAnnotation(index).empty())
            return mlir::emitError(op.getLoc())
                   << "annotations found on aggregate with no HW";
        }

        // Create ports for each non-hw field.
        for (const auto &[findex, field] : llvm::enumerate(pmi.fields)) {
          auto name =
              StringAttr::get(context, Twine(portName.strref()) + field.suffix);
          auto orientation =
              (Direction)((unsigned)portDirection ^ field.isFlip);
          PortInfo pi(name, field.type, orientation, /*symName=*/StringAttr{},
                      loc, std::nullopt, portDomain);
          newPorts.emplace_back(idxOfInsertPoint, pi);
        }
        return success();
      });
  if (failed(result))
    return failure();

  // If no new ports, we're done.
  if (newPorts.empty())
    return success();

  // Changes will be made.
  recordChanges(true);

  // Create new instance op with desired ports.

  // TODO: add and erase ports without intermediate + various array attributes.
  auto tempOp = op.cloneWithInsertedPorts(newPorts);
  opsToErase.push_back(tempOp);
  auto newInst = tempOp.cloneWithErasedPorts(portsToErase);

  auto mappingResult = walkMappings(
      portMappings, /*includeErased=*/false,
      [&](auto index, MappingInfo &pmi, auto newIndex) {
        // Identity means index -> newIndex.
        auto oldResult = op.getResult(index);
        if (pmi.identity) {
          // (Just do the RAUW here instead of tracking the mapping for this
          // too.)
          assert(oldResult.getType() == newInst.getType(newIndex));
          oldResult.replaceAllUsesWith(newInst.getResult(newIndex));
          return success();
        }

        // Create mappings for updating open aggregate users.
        auto newPortIndex = newIndex;
        if (pmi.hwType)
          hwOnlyAggMap[oldResult] = newInst.getResult(newPortIndex++);

        for (auto &field : pmi.fields) {
          auto ref = FieldRef(oldResult, field.fieldID);
          auto newVal = newInst.getResult(newPortIndex++);
          assert(newVal.getType() == field.type);
          nonHWValues[ref] = newVal;
        }
        for (auto fieldID : pmi.mapToNullInteriors) {
          auto ref = FieldRef(oldResult, fieldID);
          assert(!nonHWValues.count(ref));
          nonHWValues[ref] = {};
        }
        return success();
      });
  if (failed(mappingResult))
    return failure();

  opsToErase.push_back(op);

  return success();
}

LogicalResult Visitor::visitDecl(WireOp op) {
  auto pmi = mapType(op.getResultTypes()[0], op.getLoc(), op.getInnerSymAttr());
  if (failed(pmi))
    return failure();
  MappingInfo mappings = *pmi;

  LLVM_DEBUG({
    llvm::dbgs().indent(6) << "- wire:\n";
    llvm::dbgs().indent(10) << "name: " << op.getNameAttr() << "\n";
    llvm::dbgs().indent(10) << "type: " << op.getType(0) << "\n";
    llvm::dbgs().indent(12) << "mapping:\n";
    mappings.print(llvm::dbgs(), 14);
    llvm::dbgs() << "\n";
  });

  if (mappings.identity)
    return success();

  // Changes will be made.
  recordChanges(true);

  ImplicitLocOpBuilder builder(op.getLoc(), op);

  if (!op.getAnnotations().empty())
    return mlir::emitError(op.getLoc())
           << "annotations on open aggregates not handled yet";

  // Create the new HW wire.
  if (mappings.hwType)
    hwOnlyAggMap[op.getResult()] =
        WireOp::create(builder, mappings.hwType, op.getName(), op.getNameKind(),
                       op.getAnnotations(), mappings.newSym, op.getForceable())
            .getResult();

  // Create the non-HW wires.  Non-HW wire names are always droppable.
  for (auto &[type, fieldID, _, suffix] : mappings.fields)
    nonHWValues[FieldRef(op.getResult(), fieldID)] =
        WireOp::create(builder, type,
                       builder.getStringAttr(Twine(op.getName()) + suffix),
                       NameKindEnum::DroppableName)
            .getResult();

  for (auto fieldID : mappings.mapToNullInteriors)
    nonHWValues[FieldRef(op.getResult(), fieldID)] = {};

  opsToErase.push_back(op);

  return success();
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

FailureOr<MappingInfo> Visitor::mapType(Type type, Location errorLoc,
                                        hw::InnerSymAttr sym) {
  MappingInfo pi{false, {}, {}, {}};
  auto ftype = type_dyn_cast<FIRRTLType>(type);
  // Anything that isn't an open aggregates is left alone.
  if (!ftype || !isa<OpenBundleType, OpenVectorType>(ftype)) {
    pi.identity = true;
    return pi;
  }

  SmallVector<hw::InnerSymPropertiesAttr> newProps;

  // NOLINTBEGIN(misc-no-recursion)
  auto recurse = [&](auto &&f, FIRRTLType type, const Twine &suffix = "",
                     bool flip = false, uint64_t fieldID = 0,
                     uint64_t newFieldID = 0) -> FailureOr<FIRRTLBaseType> {
    auto newType =
        TypeSwitch<FIRRTLType, FailureOr<FIRRTLBaseType>>(type)
            .Case<FIRRTLBaseType>([](auto base) { return base; })
            .template Case<OpenBundleType>([&](OpenBundleType obTy)
                                               -> FailureOr<FIRRTLBaseType> {
              SmallVector<BundleType::BundleElement> hwElements;
              uint64_t id = 0;
              for (const auto &[index, element] :
                   llvm::enumerate(obTy.getElements())) {
                auto base =
                    f(f, element.type, suffix + "_" + element.name.strref(),
                      flip ^ element.isFlip, fieldID + obTy.getFieldID(index),
                      newFieldID + id + 1);
                if (failed(base))
                  return failure();
                if (*base) {
                  hwElements.emplace_back(element.name, element.isFlip, *base);
                  id += hw::FieldIdImpl::getMaxFieldID(*base) + 1;
                }
              }

              if (hwElements.empty()) {
                pi.mapToNullInteriors.push_back(fieldID);
                return FIRRTLBaseType{};
              }

              return BundleType::get(context, hwElements, obTy.isConst());
            })
            .template Case<OpenVectorType>([&](OpenVectorType ovTy)
                                               -> FailureOr<FIRRTLBaseType> {
              uint64_t id = 0;
              FIRRTLBaseType convert;
              // Walk for each index to extract each leaf separately, but expect
              // same hw-only type for all.
              for (auto idx : llvm::seq<size_t>(0U, ovTy.getNumElements())) {
                auto hwElementType =
                    f(f, ovTy.getElementType(), suffix + "_" + Twine(idx), flip,
                      fieldID + ovTy.getFieldID(idx), newFieldID + id + 1);
                if (failed(hwElementType))
                  return failure();
                assert((!convert || convert == *hwElementType) &&
                       "expected same hw type for all elements");
                convert = *hwElementType;
                if (convert)
                  id += hw::FieldIdImpl::getMaxFieldID(convert) + 1;
              }

              if (!convert) {
                pi.mapToNullInteriors.push_back(fieldID);
                return FIRRTLBaseType{};
              }

              return FVectorType::get(convert, ovTy.getNumElements(),
                                      ovTy.isConst());
            })
            .template Case<RefType>([&](RefType ref) {
              auto f = NonHWField{ref, fieldID, flip, {}};
              suffix.toVector(f.suffix);
              pi.fields.emplace_back(std::move(f));
              return FIRRTLBaseType{};
            })
            // This is identical to the RefType case above, but copied out
            // to try to fix a bug when combining + auto w/MSVC.
            .template Case<PropertyType>([&](PropertyType prop) {
              auto f = NonHWField{prop, fieldID, flip, {}};
              suffix.toVector(f.suffix);
              pi.fields.emplace_back(std::move(f));
              return FIRRTLBaseType{};
            })
            .Default([&](auto _) {
              pi.mapToNullInteriors.push_back(fieldID);
              return FIRRTLBaseType{};
            });
    if (failed(newType))
      return failure();

    // If there's a symbol on this, add it with adjusted fieldID.
    if (sym)
      if (auto symOnThis = sym.getSymIfExists(fieldID)) {
        if (!*newType)
          return mlir::emitError(errorLoc, "inner symbol ")
                 << symOnThis << " mapped to non-HW type";
        newProps.push_back(hw::InnerSymPropertiesAttr::get(
            context, symOnThis, newFieldID,
            StringAttr::get(context, "public")));
      }
    return newType;
  };

  auto hwType = recurse(recurse, ftype);
  if (failed(hwType))
    return failure();
  pi.hwType = *hwType;

  assert(pi.hwType != type);
  // NOLINTEND(misc-no-recursion)

  if (sym) {
    assert(sym.size() == newProps.size());

    if (!pi.hwType && !newProps.empty())
      return mlir::emitError(errorLoc, "inner symbol on non-HW type");

    llvm::sort(newProps, [](auto &p, auto &q) {
      return p.getFieldID() < q.getFieldID();
    });
    pi.newSym = hw::InnerSymAttr::get(context, newProps);
  }

  return pi;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerOpenAggsPass
    : public circt::firrtl::impl::LowerOpenAggsBase<LowerOpenAggsPass> {
  LowerOpenAggsPass() = default;
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void LowerOpenAggsPass::runOnOperation() {
  CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);

  SmallVector<Operation *, 0> ops(getOperation().getOps<FModuleLike>());

  LLVM_DEBUG(llvm::dbgs() << "Visiting modules:\n");
  std::atomic<bool> madeChanges = false;
  auto result = failableParallelForEach(&getContext(), ops, [&](Operation *op) {
    Visitor visitor(&getContext());
    auto result = visitor.visit(cast<FModuleLike>(op));
    if (visitor.madeChanges())
      madeChanges = true;
    return result;
  });

  if (result.failed())
    signalPassFailure();
  if (!madeChanges)
    markAllAnalysesPreserved();
}
