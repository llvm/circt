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
static LogicalResult emit(TclOutputState &s, RootedInstancePathAttr path,
                          Operation *op, StringRef attrKey,
                          PhysLocationAttr pla) {

  if (!attrKey.startswith_insensitive("loc:"))
    return op->emitError("Error in '")
           << attrKey << "' PhysLocation attribute. Expected loc:<entityName>.";

  StringRef childEntity = attrKey.substr(4);
  if (childEntity.empty())
    return op->emitError("Entity name cannot be empty in 'loc:<entityName>'");

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
  emitPath(s, path);
  // If instance name is specified, add it in between the parent entity path and
  // the child entity patch.
  if (auto inst = dyn_cast<hw::InstanceOp>(op))
    s.os << inst.instanceName() << '|';
  else if (auto name = op->getAttrOfType<StringAttr>("name"))
    s.os << name.getValue() << '|';
  s.os << childEntity << '\n';

  return success();
}

static LogicalResult emitAttr(TclOutputState &s, RootedInstancePathAttr path,
                              Operation *op, StringRef attrName,
                              Attribute attr) {
  if (auto loc = attr.dyn_cast<PhysLocationAttr>())
    if (failed(emit(s, path, op, attrName, loc)))
      return failure();
  return success();
}

/// Export the TCL for a particular entity, corresponding to op. Do this
/// recusively, assume that all descendants are in the same entity. When this is
/// no longer a sound assuption, we'll have to refactor this code. For now, only
/// HWModule instances create a new entity.
static LogicalResult emitAttrs(TclOutputState &s, Operation *root,
                               Operation *op) {

  auto rootName = FlatSymbolRefAttr::get(root->getContext(),
                                         SymbolTable::getSymbolName(root));

  // Iterate again through the attributes, looking for instance-specific
  // attributes.
  for (NamedAttribute attr : op->getAttrs()) {
    auto result =
        llvm::TypeSwitch<Attribute, LogicalResult>(attr.second)

            // Handle switch instance.
            .Case([&](SwitchInstanceAttr instSwitch) {
              for (auto switchCase : instSwitch.getCases()) {
                // Filter for only paths rooted at the root module.
                auto caseRoot = switchCase.getInst().getRootModule();
                if (caseRoot != rootName)
                  continue;

                // Output the attribute.
                if (failed(emitAttr(s, switchCase.getInst(), op, attr.first,
                                    switchCase.getAttr())))
                  return failure();
              }
              return success();
            })

            // Physloc outside of a switch instance is not valid.
            .Case([op](PhysLocationAttr) {
              return op->emitOpError("PhysLoc attribute must be inside an "
                                     "instance switch attribute");
            })

            // Ignore attributes we don't understand.
            .Default([](Attribute) { return success(); });

    if (failed(result))
      return failure();
  }
  return success();
}

/// Write out all the relevant tcl commands. Create one 'proc' per module which
/// takes the parent entity name since we don't assume that the created module
/// is the top level for the entire design.
LogicalResult circt::msft::exportQuartusTcl(hw::HWModuleOp hwMod,
                                            llvm::raw_ostream &os) {
  TclOutputState state(os);
  mlir::ModuleOp mlirModule = hwMod->getParentOfType<mlir::ModuleOp>();

  os << "proc " << hwMod.getName() << "_config { parent } {\n";

  auto result = mlirModule->walk([&](Operation *innerOp) {
    if (failed(emitAttrs(state, hwMod, innerOp)))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });

  os << "}\n\n";
  return failure(result.wasInterrupted());
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
