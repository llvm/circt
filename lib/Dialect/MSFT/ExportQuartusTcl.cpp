//===- ExportQuartusTcl.cpp - Emit Quartus-flavored Tcl -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Write out Tcl with the appropriate API calls for Quartus.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/DeviceDB.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace hw;
using namespace msft;

TclEmitter::TclEmitter(mlir::ModuleOp topLevel)
    : topLevel(topLevel), populated(false) {}

LogicalResult TclEmitter::populate() {
  if (populated)
    return success();

  // Populated the symbol cache.
  for (auto symOp : topLevel.getOps<mlir::SymbolOpInterface>())
    if (auto name = symOp.getNameAttr())
      topLevelSymbols.addDefinition(name, symOp);
  topLevelSymbols.freeze();
  populated = true;

  // Bin any operations we may need to emit based on the root module in the
  // instance hierarchy path and the potential instance name.

  // Look in InstanceHierarchyOps to get the instance named ones.
  for (auto hier : topLevel.getOps<InstanceHierarchyOp>()) {
    Operation *mod = topLevelSymbols.getDefinition(hier.getTopModuleRefAttr());
    auto &tclOps = tclOpsForModInstance[mod][hier.getInstNameAttr()];
    for (auto tclOp : hier.getOps<DynInstDataOpInterface>()) {
      assert(tclOp.getTopModule(topLevelSymbols) == mod &&
             "Referenced mod does does not match");
      tclOps.push_back(tclOp);
    }
  }

  // Locations at the global scope are assumed to refer to the module without an
  // instance.
  for (auto tclOp : topLevel.getOps<DynInstDataOpInterface>()) {
    Operation *mod = tclOp.getTopModule(topLevelSymbols);
    assert(mod && "Must be able to resolve top module");
    tclOpsForModInstance[mod][{}].push_back(tclOp);
  }

  return success();
}

Operation *TclEmitter::getDefinition(FlatSymbolRefAttr sym) {
  if (failed(populate()))
    return nullptr;
  return topLevelSymbols.getDefinition(sym);
}

// TODO: Currently assumes Stratix 10 and QuartusPro. Make more general.
namespace {
/// Utility struct to assist in output and track other relevent state which are
/// not specific to the entity hierarchy (global WRT to the entity hierarchy).
struct TclOutputState {
  TclOutputState(TclEmitter &emitter, llvm::raw_ostream &os)
      : os(os), emitter(emitter) {}

  llvm::raw_ostream &os;
  llvm::raw_ostream &indent() {
    os.indent(2);
    return os;
  };

  TclEmitter &emitter;
  SmallVector<Attribute> symbolRefs;

  void emit(PhysLocationAttr);
  LogicalResult emitLocationAssignment(UnaryDynInstDataOpInterface refOp,
                                       PhysLocationAttr,
                                       std::optional<StringRef> subpath);

  LogicalResult emit(PDPhysRegionOp region);
  LogicalResult emit(PDPhysLocationOp loc);
  LogicalResult emit(PDRegPhysLocationOp);
  LogicalResult emit(DynamicInstanceVerbatimAttrOp attr);
  LogicalResult emit(PDMulticycleOp op);

  void emitPath(hw::HierPathOp ref, std::optional<StringRef> subpath);
  void emitInnerRefPart(hw::InnerRefAttr innerRef);

  /// Get the HierPathOp to which the given operation is pointing. Add it to
  /// the set of used global refs.
  HierPathOp getRefOp(UnaryDynInstDataOpInterface op) {
    return getRefOp(op.getLoc(), op.getPathSym());
  }

  /// Get the HierPathOp to which a given value is pointing. Add it to the
  /// set of used global refs.
  HierPathOp getRefOp(Location loc, FlatSymbolRefAttr pathSym) {
    auto ref = dyn_cast_or_null<hw::HierPathOp>(emitter.getDefinition(pathSym));
    if (ref)
      emitter.usedRef(ref);
    else
      emitError(loc, "could not find hw.hierpath named ") << pathSym;
    return ref;
  }
};
} // anonymous namespace

void TclOutputState::emitInnerRefPart(hw::InnerRefAttr innerRef) {
  // We append new symbolRefs to the state, so s.symbolRefs.size() is the
  // index of the InnerRefAttr we are about to add.
  os << "{{" << symbolRefs.size() << "}}";

  // Append a new inner reference for the template above.
  symbolRefs.push_back(innerRef);
}

void TclOutputState::emitPath(hw::HierPathOp ref,
                              std::optional<StringRef> subpath) {
  os << "{{" << symbolRefs.size() << ":|}}";
  symbolRefs.push_back(FlatSymbolRefAttr::get(ref));
  if (subpath)
    os << subpath;
}

void TclOutputState::emit(PhysLocationAttr pla) {
  // Different devices have different 'number' letters (the 'N' in 'N0'). M20Ks
  // and DSPs happen to have the same one, probably because they never co-exist
  // at the same location.
  char numCharacter;
  switch (pla.getPrimitiveType().getValue()) {
  case PrimitiveType::M20K:
    os << "M20K";
    numCharacter = 'N';
    break;
  case PrimitiveType::DSP:
    os << "MPDSP";
    numCharacter = 'N';
    break;
  case PrimitiveType::FF:
    os << "FF";
    numCharacter = 'N';
    break;
  }

  // Write out the rest of the location info.
  os << "_X" << pla.getX() << "_Y" << pla.getY() << "_" << numCharacter
     << pla.getNum();
}

/// Emit tcl in the form of:
/// "set_location_assignment MPDSP_X34_Y285_N0 -to
/// $parent|fooInst|entityName(subpath)"
LogicalResult
TclOutputState::emitLocationAssignment(UnaryDynInstDataOpInterface refOp,
                                       PhysLocationAttr loc,
                                       std::optional<StringRef> subpath) {
  indent() << "set_location_assignment ";
  emit(loc);

  // To which entity does this apply?
  os << " -to $parent|";
  emitPath(getRefOp(refOp), subpath);

  return success();
}

LogicalResult TclOutputState::emit(PDPhysLocationOp loc) {
  if (failed(emitLocationAssignment(loc, loc.getLoc(), loc.getSubPath())))
    return failure();
  os << '\n';
  return success();
}

LogicalResult TclOutputState::emit(PDRegPhysLocationOp locs) {
  ArrayRef<PhysLocationAttr> locArr = locs.getLocs().getLocs();
  for (size_t i = 0, e = locArr.size(); i < e; ++i) {
    PhysLocationAttr pla = locArr[i];
    if (!pla)
      continue;
    if (failed(emitLocationAssignment(locs, pla, {})))
      return failure();
    os << "[" << i << "]\n";
  }
  return success();
}

LogicalResult TclOutputState::emit(PDMulticycleOp op) {
  indent() << "set_multicycle_path ";
  os << "-hold 1 ";
  os << "-setup " << op.getCycles() << " ";
  os << "-from [get_registers {$parent|";
  emitPath(getRefOp(op.getLoc(), op.getSourceAttr()), std::nullopt);
  os << "}] ";
  os << "-to [get_registers {$parent|";
  emitPath(getRefOp(op.getLoc(), op.getDestAttr()), std::nullopt);
  os << "}]\n";
  return success();
}

/// Emit tcl in the form of:
/// "set_global_assignment -name NAME VALUE -to $parent|fooInst|entityName"
LogicalResult TclOutputState::emit(DynamicInstanceVerbatimAttrOp attr) {
  HierPathOp ref = getRefOp(attr);
  indent() << "set_instance_assignment -name " << attr.getName() << " "
           << attr.getValue();

  // To which entity does this apply?
  os << " -to $parent|";
  emitPath(ref, attr.getSubPath());
  os << '\n';
  return success();
}

/// Emit tcl in the form of:
/// set_instance_assignment -name PLACE_REGION "X1 Y1 X20 Y20" -to $parent|a|b|c
/// set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|a|b|c
/// set_instance_assignment -name CORE_ONLY_PLACE_REGION ON -to $parent|a|b|c
/// set_instance_assignment -name REGION_NAME test_region -to $parent|a|b|c
LogicalResult TclOutputState::emit(PDPhysRegionOp region) {
  HierPathOp ref = getRefOp(region);

  auto physicalRegion = dyn_cast_or_null<DeclPhysicalRegionOp>(
      emitter.getDefinition(region.getPhysRegionRefAttr()));
  if (!physicalRegion)
    return region.emitOpError(
               "could not find physical region declaration named ")
           << region.getPhysRegionRefAttr();

  // PLACE_REGION directive.
  indent() << "set_instance_assignment -name PLACE_REGION \"";
  auto physicalBounds =
      physicalRegion.getBounds().getAsRange<PhysicalBoundsAttr>();
  llvm::interleave(
      physicalBounds, os,
      [&](PhysicalBoundsAttr bounds) {
        os << 'X' << bounds.getXMin() << ' ';
        os << 'Y' << bounds.getYMin() << ' ';
        os << 'X' << bounds.getXMax() << ' ';
        os << 'Y' << bounds.getYMax();
      },
      ";");
  os << '"';

  os << " -to $parent|";
  emitPath(ref, region.getSubPath());
  os << '\n';

  // RESERVE_PLACE_REGION directive.
  indent() << "set_instance_assignment -name RESERVE_PLACE_REGION OFF";
  os << " -to $parent|";
  emitPath(ref, region.getSubPath());
  os << '\n';

  // CORE_ONLY_PLACE_REGION directive.
  indent() << "set_instance_assignment -name CORE_ONLY_PLACE_REGION ON";
  os << " -to $parent|";
  emitPath(ref, region.getSubPath());
  os << '\n';

  // REGION_NAME directive.
  indent() << "set_instance_assignment -name REGION_NAME ";
  os << physicalRegion.getName();
  os << " -to $parent|";
  emitPath(ref, region.getSubPath());
  os << '\n';
  return success();
}

/// Write out all the relevant tcl commands. Create one 'proc' per module which
/// takes the parent entity name since we don't assume that the created module
/// is the top level for the entire design.
LogicalResult TclEmitter::emit(Operation *hwMod, StringRef outputFile) {
  if (failed(populate()))
    return failure();

  // Build up the output Tcl, tracking symbol references in state.
  std::string s;
  llvm::raw_string_ostream os(s);
  TclOutputState state(*this, os);

  // Iterate through all the "instances" for 'hwMod' and produce a tcl proc for
  // each one.
  for (const auto &tclOpsForInstancesKV : tclOpsForModInstance[hwMod]) {
    StringAttr instName = tclOpsForInstancesKV.first;
    os << "proc {{" << state.symbolRefs.size() << "}}";
    if (instName)
      os << '_' << instName.getValue();
    os << "_config { parent } {\n";
    state.symbolRefs.push_back(SymbolRefAttr::get(hwMod));

    // Loop through the ops relevant to the specified root module "instance".
    LogicalResult ret = success();
    const auto &tclOpsForMod = tclOpsForInstancesKV.second;
    for (Operation *tclOp : tclOpsForMod) {
      LogicalResult rc =
          TypeSwitch<Operation *, LogicalResult>(tclOp)
              .Case([&](PDPhysLocationOp op) { return state.emit(op); })
              .Case([&](PDRegPhysLocationOp op) { return state.emit(op); })
              .Case([&](PDPhysRegionOp op) { return state.emit(op); })
              .Case([&](PDMulticycleOp op) { return state.emit(op); })
              .Case([&](DynamicInstanceVerbatimAttrOp op) {
                return state.emit(op);
              })
              .Default([](Operation *op) {
                return op->emitOpError("could not determine how to output tcl");
              });
      if (failed(rc))
        ret = failure();
    }
    os << "}\n\n";
  }

  // Create a verbatim op containing the Tcl and symbol references.
  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      UnknownLoc::get(hwMod->getContext()), hwMod->getBlock());
  emit::FileOp::create(builder, outputFile, [&] {
    sv::VerbatimOp::create(builder, os.str(), ValueRange{},
                           builder.getArrayAttr(state.symbolRefs));
  });

  return success();
}
