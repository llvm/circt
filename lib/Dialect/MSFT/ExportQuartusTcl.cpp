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
  // instance hierarchy path.
  for (auto tclOp : topLevel.getOps<DynInstDataOpInterface>()) {
    FlatSymbolRefAttr refSym = tclOp.getGlobalRefSym();
    if (!refSym)
      return tclOp->emitOpError("must run dynamic instance lowering first");
    auto ref = dyn_cast_or_null<hw::GlobalRefOp>(
        topLevelSymbols.getDefinition(refSym));
    if (!ref)
      return tclOp->emitOpError("could not find hw.globalRef ") << refSym;
    if (ref.namepath().empty())
      continue;
    auto modSym = FlatSymbolRefAttr::get(
        ref.namepath()[0].cast<hw::InnerRefAttr>().getModule());
    Operation *mod = topLevelSymbols.getDefinition(modSym);
    assert(mod &&
           "Invalid IR -- should have been caught by GlobalRef verifier");
    tclOpsForMod[mod].push_back(tclOp);
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

  LogicalResult emit(PDPhysRegionOp region);
  LogicalResult emit(PDPhysLocationOp loc);

  void emitPath(hw::GlobalRefOp ref, Optional<StringRef> subpath);
  void emitInnerRefPart(hw::InnerRefAttr innerRef);
};
} // anonymous namespace

void TclOutputState::emitInnerRefPart(hw::InnerRefAttr innerRef) {
  // We append new symbolRefs to the state, so s.symbolRefs.size() is the
  // index of the InnerRefAttr we are about to add.
  os << "{{" << symbolRefs.size() << "}}";

  // Append a new inner reference for the template above.
  symbolRefs.push_back(innerRef);
}

void TclOutputState::emitPath(hw::GlobalRefOp ref,
                              Optional<StringRef> subpath) {
  // Traverse each part of the path.
  auto parts = ref.namepathAttr().getAsRange<hw::InnerRefAttr>();
  auto lastPart = std::prev(parts.end());
  for (auto part : parts) {
    emitInnerRefPart(part);
    if (part != *lastPart)
      os << '|';
  }

  // Some placements don't require subpaths.
  if (subpath)
    os << '|' << subpath;

  os << '\n';
}

/// Emit tcl in the form of:
/// "set_location_assignment MPDSP_X34_Y285_N0 -to $parent|fooInst|entityName"
LogicalResult TclOutputState::emit(PDPhysLocationOp loc) {

  auto ref =
      dyn_cast_or_null<hw::GlobalRefOp>(emitter.getDefinition(loc.refAttr()));
  if (!ref)
    return loc.emitOpError("could not find hw.globalRef named ")
           << loc.refAttr();
  PhysLocationAttr pla = loc.loc();
  indent() << "set_location_assignment ";

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

  // To which entity does this apply?
  os << " -to $parent|";
  emitPath(ref, loc.subPath());
  return success();
}

/// Emit tcl in the form of:
/// set_instance_assignment -name PLACE_REGION "X1 Y1 X20 Y20" -to $parent|a|b|c
/// set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|a|b|c
/// set_instance_assignment -name CORE_ONLY_PLACE_REGION ON -to $parent|a|b|c
/// set_instance_assignment -name REGION_NAME test_region -to $parent|a|b|c
LogicalResult TclOutputState::emit(PDPhysRegionOp region) {
  auto ref = dyn_cast_or_null<hw::GlobalRefOp>(
      emitter.getDefinition(region.refAttr()));
  if (!ref)
    return region.emitOpError("could not find hw.globalRef named ")
           << region.refAttr();

  auto physicalRegion = dyn_cast_or_null<DeclPhysicalRegionOp>(
      emitter.getDefinition(region.physRegionRefAttr()));
  if (!physicalRegion)
    return region.emitOpError(
               "could not find physical region declaration named ")
           << region.physRegionRefAttr();

  // PLACE_REGION directive.
  indent() << "set_instance_assignment -name PLACE_REGION \"";
  auto physicalBounds =
      physicalRegion.bounds().getAsRange<PhysicalBoundsAttr>();
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
  emitPath(ref, region.subPath());

  // RESERVE_PLACE_REGION directive.
  indent() << "set_instance_assignment -name RESERVE_PLACE_REGION OFF";
  os << " -to $parent|";
  emitPath(ref, region.subPath());

  // CORE_ONLY_PLACE_REGION directive.
  indent() << "set_instance_assignment -name CORE_ONLY_PLACE_REGION ON";
  os << " -to $parent|";
  emitPath(ref, region.subPath());

  // REGION_NAME directive.
  indent() << "set_instance_assignment -name REGION_NAME ";
  os << physicalRegion.getName();
  os << " -to $parent|";
  emitPath(ref, region.subPath());
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
  os << "proc {{" << state.symbolRefs.size() << "}}_config { parent } {\n";
  state.symbolRefs.push_back(SymbolRefAttr::get(hwMod));

  // Loop through the ops relevant to the specified root module.
  LogicalResult ret = success();
  for (Operation *tclOp : tclOpsForMod[hwMod]) {
    LogicalResult rc =
        TypeSwitch<Operation *, LogicalResult>(tclOp)
            .Case([&](PDPhysLocationOp op) { return state.emit(op); })
            .Case([&](PDPhysRegionOp op) { return state.emit(op); })
            .Default([](Operation *op) {
              return op->emitOpError("could not determine how to output tcl");
            });
    if (failed(rc))
      ret = failure();
  }
  os << "}\n\n";

  // Create a verbatim op containing the Tcl and symbol references.
  OpBuilder builder = OpBuilder::atBlockEnd(hwMod->getBlock());
  auto verbatim = builder.create<sv::VerbatimOp>(
      builder.getUnknownLoc(), os.str(), ValueRange{},
      builder.getArrayAttr(state.symbolRefs));

  // When requested, give the verbatim op an output file.
  if (!outputFile.empty()) {
    auto outputFileAttr =
        OutputFileAttr::getFromFilename(builder.getContext(), outputFile);
    verbatim->setAttr("output_file", outputFileAttr);
  }

  return success();
}
