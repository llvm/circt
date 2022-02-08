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
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/DeviceDB.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace hw;
using namespace msft;

// TODO: Currently assumes Stratix 10 and QuartusPro. Make more general.

namespace {
/// Utility struct to assist in output and track other relevent state which are
/// not specific to the entity hierarchy (global WRT to the entity hierarchy).
struct TclOutputState {
  TclOutputState(llvm::raw_ostream &os) : os(os) {}

  llvm::raw_ostream &os;
  llvm::raw_ostream &indent() {
    os.indent(2);
    return os;
  };

  SmallVector<Attribute> symbolRefs;
};
} // anonymous namespace

void emitInnerRefPart(TclOutputState &s, hw::InnerRefAttr innerRef) {
  // We append new symbolRefs to the state, so s.symbolRefs.size() is the
  // index of the InnerRefAttr we are about to add.
  s.os << "{{" << s.symbolRefs.size() << "}}";

  // Append a new inner reference for the template above.
  s.symbolRefs.push_back(innerRef);
}

void emitPath(TclOutputState &s, PlacementDB::PlacedInstance inst) {
  // Traverse each part of the path.
  auto parts = inst.path.getAsRange<hw::InnerRefAttr>();
  auto lastPart = std::prev(parts.end());
  for (auto part : parts) {
    emitInnerRefPart(s, part);
    if (part != *lastPart)
      s.os << '|';
  }

  // Some placements don't require subpaths.
  if (!inst.subpath.empty()) {
    s.os << '|';
    s.os << inst.subpath;
  }

  s.os << '\n';
}

/// Emit tcl in the form of:
/// "set_location_assignment MPDSP_X34_Y285_N0 -to $parent|fooInst|entityName"
static void emit(TclOutputState &s, PlacementDB::PlacedInstance inst,
                 PhysLocationAttr pla) {

  s.indent() << "set_location_assignment ";

  // Different devices have different 'number' letters (the 'N' in 'N0'). M20Ks
  // and DSPs happen to have the same one, probably because they never co-exist
  // at the same location.
  char numCharacter;
  switch (pla.getPrimitiveType().getValue()) {
  case PrimitiveType::M20K:
    s.os << "M20K";
    numCharacter = 'N';
    break;
  case PrimitiveType::DSP:
    s.os << "MPDSP";
    numCharacter = 'N';
    break;
  case PrimitiveType::FF:
    s.os << "FF";
    numCharacter = 'N';
    break;
  }

  // Write out the rest of the location info.
  s.os << "_X" << pla.getX() << "_Y" << pla.getY() << "_" << numCharacter
       << pla.getNum();

  // To which entity does this apply?
  s.os << " -to $parent|";
  emitPath(s, inst);
}

/// Emit tcl in the form of:
/// set_instance_assignment -name PLACE_REGION "X1 Y1 X20 Y20" -to $parent|a|b|c
/// set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|a|b|c
/// set_instance_assignment -name CORE_ONLY_PLACE_REGION ON -to $parent|a|b|c
/// set_instance_assignment -name REGION_NAME test_region -to $parent|a|b|c
static void emit(TclOutputState &s, PlacementDB::PlacedInstance inst,
                 PhysicalRegionRefAttr regionRef) {
  auto topModule = inst.op->getParentOfType<mlir::ModuleOp>();
  auto physicalRegion =
      topModule.lookupSymbol<PhysicalRegionOp>(regionRef.getPhysicalRegion());
  assert(physicalRegion && "must reference an existant physical region");

  // PLACE_REGION directive.
  s.indent() << "set_instance_assignment -name PLACE_REGION \"";
  auto physicalBounds =
      physicalRegion.bounds().getAsRange<PhysicalBoundsAttr>();
  llvm::interleave(
      physicalBounds, s.os,
      [&s](PhysicalBoundsAttr bounds) {
        s.os << 'X' << bounds.getXMin() << ' ';
        s.os << 'Y' << bounds.getYMin() << ' ';
        s.os << 'X' << bounds.getXMax() << ' ';
        s.os << 'Y' << bounds.getYMax();
      },
      ";");
  s.os << '"';
  s.os << " -to $parent|";
  emitPath(s, inst);

  // RESERVE_PLACE_REGION directive.
  s.indent() << "set_instance_assignment -name RESERVE_PLACE_REGION OFF";
  s.os << " -to $parent|";
  emitPath(s, inst);

  // CORE_ONLY_PLACE_REGION directive.
  s.indent() << "set_instance_assignment -name CORE_ONLY_PLACE_REGION ON";
  s.os << " -to $parent|";
  emitPath(s, inst);

  // REGION_NAME directive.
  s.indent() << "set_instance_assignment -name REGION_NAME ";
  s.os << physicalRegion.getName();
  s.os << " -to $parent|";
  emitPath(s, inst);
}

static bool isEntityExtern(Operation *op) {
  auto globalRef = cast<hw::GlobalRefOp>(op);
  auto path = globalRef.namepath().getValue();
  auto leafRef = path.back().dyn_cast<FlatSymbolRefAttr>();
  if (!leafRef)
    return false;

  auto rootMod = op->getParentOfType<mlir::ModuleOp>();
  auto *leafRefOp = rootMod.lookupSymbol(leafRef);
  return isa<EntityExternOp>(leafRefOp);
}

/// Write out all the relevant tcl commands. Create one 'proc' per module which
/// takes the parent entity name since we don't assume that the created module
/// is the top level for the entire design.
LogicalResult circt::msft::exportQuartusTcl(MSFTModuleOp hwMod,
                                            StringRef outputFile) {
  // Build up the output Tcl, tracking symbol references in state.
  std::string s;
  llvm::raw_string_ostream os(s);
  TclOutputState state(os);
  PlacementDB db(hwMod);
  size_t failures = db.addDesignPlacements();
  if (failures != 0)
    return hwMod->emitError("Could not place ") << failures << " instances";

  os << "proc {{" << state.symbolRefs.size() << "}}_config { parent } {\n";
  state.symbolRefs.push_back(SymbolRefAttr::get(hwMod));

  db.walkPlacements(
      [&state](PhysLocationAttr loc, PlacementDB::PlacedInstance inst) {
        // Skip entities which we don't need to emit placements for.
        if (isEntityExtern(inst.op))
          return;
        emit(state, inst, loc);
      });

  db.walkRegionPlacements([&state](PhysicalRegionRefAttr regionRef,
                                   PlacementDB::PlacedInstance inst) {
    // Skip entities which we don't need to emit placements for.
    if (isEntityExtern(inst.op))
      return;
    emit(state, inst, regionRef);
  });

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
