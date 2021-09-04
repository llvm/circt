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

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/MSFT/DeviceDB.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Support/LLVM.h"
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
};
} // anonymous namespace

void emitPath(TclOutputState &s, RootedInstancePathAttr path) {
  for (auto part : path.getPath())
    s.os << part.getValue() << '|';
}

/// Emit tcl in the form of:
/// "set_location_assignment MPDSP_X34_Y285_N0 -to $parent|fooInst|entityName"
static void emit(TclOutputState &s, DeviceDB::PlacedInstance inst,
                 PhysLocationAttr pla) {

  s.indent() << "set_location_assignment ";

  // Different devices have different 'number' letters (the 'N' in 'N0'). M20Ks
  // and DSPs happen to have the same one, probably because they never co-exist
  // at the same location.
  char numCharacter;
  switch (pla.getDevType().getValue()) {
  case DeviceType::M20K:
    s.os << "M20K";
    numCharacter = 'N';
    break;
  case DeviceType::DSP:
    s.os << "MPDSP";
    numCharacter = 'N';
    break;
  }

  // Write out the rest of the location info.
  s.os << "_X" << pla.getX() << "_Y" << pla.getY() << "_" << numCharacter
       << pla.getNum();

  // To which entity does this apply?
  s.os << " -to $parent|";
  emitPath(s, inst.path);
  // If instance name is specified, add it in between the parent entity path and
  // the child entity patch.
  if (auto instOp = dyn_cast<hw::InstanceOp>(inst.op))
    s.os << instOp.instanceName() << '|';
  else if (auto name = inst.op->getAttrOfType<StringAttr>("name"))
    s.os << name.getValue() << '|';
  s.os << inst.subpath << '\n';
}

/// Write out all the relevant tcl commands. Create one 'proc' per module which
/// takes the parent entity name since we don't assume that the created module
/// is the top level for the entire design.
LogicalResult circt::msft::exportQuartusTcl(hw::HWModuleOp hwMod,
                                            llvm::raw_ostream &os) {
  TclOutputState state(os);
  DeviceDB db(hwMod.getContext(), hwMod);
  size_t failures = db.addDesignPlacements();
  if (failures != 0)
    return hwMod->emitError("Could not place ") << failures << " instances";

  os << "proc " << hwMod.getName() << "_config { parent } {\n";

  db.walkPlacements(
      [&state](PhysLocationAttr loc, DeviceDB::PlacedInstance inst) {
        emit(state, inst, loc);
      });

  os << "}\n\n";
  return success();
}

static LogicalResult exportQuartusTclForAll(mlir::ModuleOp mod,
                                            llvm::raw_ostream &os) {
  for (Operation &op : mod.getBody()->getOperations()) {
    if (auto hwmod = dyn_cast<hw::HWModuleOp>(op))
      if (failed(exportQuartusTcl(hwmod, os)))
        return failure();
  }
  return success();
}

void circt::msft::registerMSFTTclTranslation() {
  mlir::TranslateFromMLIRRegistration toQuartusTcl(
      "export-quartus-tcl", exportQuartusTclForAll,
      [](mlir::DialectRegistry &registry) {
        registry.insert<MSFTDialect, HWDialect>();
      });
}
