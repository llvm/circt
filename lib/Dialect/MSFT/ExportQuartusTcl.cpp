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
#include "circt/Dialect/MSFT/DeviceDB.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/SV/SVOps.h"
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

void emitPath(TclOutputState &s, RootedInstancePathAttr path,
              SymbolCache &symCache) {
  for (auto part : path.getPath()) {
    auto inst =
        dyn_cast_or_null<msft::InstanceOp>(symCache.getDefinition(part));
    assert(inst && "path instance must be in symbol cache");
    auto mod = inst->getParentOfType<MSFTModuleOp>();
    s.os << "{{" << s.symbolRefs.size() << "}}" << '|';
    s.symbolRefs.push_back(
        InnerRefAttr::get(mod.getNameAttr(), inst.sym_nameAttr()));
  }
}

/// Emit tcl in the form of:
/// "set_location_assignment MPDSP_X34_Y285_N0 -to $parent|fooInst|entityName"
static void emit(TclOutputState &s, PlacementDB::PlacedInstance inst,
                 PhysLocationAttr pla, SymbolCache &symCache) {

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
  }

  // Write out the rest of the location info.
  s.os << "_X" << pla.getX() << "_Y" << pla.getY() << "_" << numCharacter
       << pla.getNum();

  // To which entity does this apply?
  s.os << " -to $parent|";
  emitPath(s, inst.path, symCache);
  // If instance name is specified, add it in between the parent entity path and
  // the child entity patch.
  if (auto instOp = dyn_cast<msft::InstanceOp>(inst.op))
    s.os << instOp.instanceName() << '|';
  else if (auto name = inst.op->getAttrOfType<StringAttr>("name"))
    s.os << name.getValue() << '|';
  s.os << inst.subpath << '\n';
}

/// Create a SymbolCache to use during Tcl export.
void circt::msft::populateSymbolCache(mlir::ModuleOp mod, SymbolCache &cache) {
  // Traverse each module and each instance within the module.
  for (auto hwMod : mod.getOps<MSFTModuleOp>()) {
    for (auto inst : hwMod.getOps<msft::InstanceOp>()) {
      // Use the instance symbol name.
      StringAttr symName = inst.sym_nameAttr();

      // Add the symbol to the cache.
      cache.addDefinition(symName, inst);
    }
  }

  cache.freeze();
}

/// Write out all the relevant tcl commands. Create one 'proc' per module which
/// takes the parent entity name since we don't assume that the created module
/// is the top level for the entire design.
LogicalResult circt::msft::exportQuartusTcl(MSFTModuleOp hwMod,
                                            SymbolCache &symCache) {
  // Build up the output Tcl, tracking symbol references in state.
  std::string s;
  llvm::raw_string_ostream os(s);
  TclOutputState state(os);
  PlacementDB db(hwMod);
  size_t failures = db.addDesignPlacements();
  if (failures != 0)
    return hwMod->emitError("Could not place ") << failures << " instances";

  // Don't emit empty procs if the database is empty.
  if (db.empty())
    return success();

  os << "proc " << hwMod.getName() << "_config { parent } {\n";

  db.walkPlacements([&state, &symCache](PhysLocationAttr loc,
                                        PlacementDB::PlacedInstance inst) {
    emit(state, inst, loc, symCache);
  });

  os << "}\n\n";

  // Create a verbatim op containing the Tcl and symbol references.
  OpBuilder builder = OpBuilder::atBlockEnd(hwMod->getBlock());
  auto verbatim = builder.create<sv::VerbatimOp>(
      builder.getUnknownLoc(), os.str(), ValueRange{},
      builder.getArrayAttr(state.symbolRefs));

  // Give the verbatim op an output file.
  // TODO: the filename should be a pass option.
  auto outputFile =
      OutputFileAttr::getFromFilename(builder.getContext(), "placements.tcl");
  verbatim->setAttr("output_file", outputFile);

  return success();
}
