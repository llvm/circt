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
using namespace rtl;
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

  /// Track which modules have been examined so we can issue warnings for
  /// instance-specific annotations if we write out the same one twice.
  SmallPtrSet<Operation *, 32> modulesEmitted;
};
} // anonymous namespace

namespace {
/// Represents a Verilog 'entity' -- a unique identifier which locates a
/// particular instance in the module-instance hierarchy.
struct Entity {
  Entity(TclOutputState &s)
      : s(s), parent(nullptr), name("$parent"), insideEmittedModule(false) {}
  Entity(Entity *parent, StringRef name, bool insideEmittedModule)
      : s(parent->s), parent(parent), name(name),
        insideEmittedModule(insideEmittedModule) {}

  /// Return the entity inside this instance.
  Optional<Entity> enter(InstanceOp inst);
  /// Emit a physical location tcl command.
  LogicalResult emit(Operation *, StringRef attrName, StringRef instName,
                     PhysLocationAttr);
  /// Emit the entity hierarchy.
  void emitPath();

  TclOutputState &s;
  Entity *parent;
  StringRef name;
  bool insideEmittedModule;

  StringSet<> emittedAttrKeys;
};
} // anonymous namespace

Optional<Entity> Entity::enter(InstanceOp inst) {
  auto mod = dyn_cast_or_null<rtl::RTLModuleOp>(inst.getReferencedModule());
  if (!mod) // Could be an extern module, which we should ignore.
    return {};
  bool modEmitted = insideEmittedModule ||
                    s.modulesEmitted.find(mod) != s.modulesEmitted.end();
  s.modulesEmitted.insert(mod);

  return Entity(this, inst.instanceName(), modEmitted);
}

/// Emit tcl in the form of:
/// "set_location_assignment MPDSP_X34_Y285_N0 -to $parent|fooInst|entityName"
LogicalResult Entity::emit(Operation *op, StringRef attrKey, StringRef instName,
                           PhysLocationAttr pla) {
  if (insideEmittedModule)
    op->emitWarning(
        "The placement information for this module has already been emitted. "
        "Modules are required to only be instantiated once.");

  if (!attrKey.startswith_lower("loc:"))
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
  s.os << " -to ";
  emitPath();
  // If instance name is specified, add it in between the parent entity path and
  // the child entity patch.
  if (!instName.empty())
    s.os << instName << '|';
  else if (auto name = op->getAttrOfType<StringAttr>("name"))
    s.os << name.getValue() << '|';
  s.os << childEntity << '\n';

  emittedAttrKeys.insert(attrKey);
  return success();
}

void Entity::emitPath() {
  if (parent)
    parent->emitPath();
  // Names are separated by '|'.
  s.os << name << "|";
}

/// Export the TCL for a particular entity, corresponding to op. Do this
/// recusively, assume that all descendants are in the same entity. When this is
/// no longer a sound assuption, we'll have to refactor this code. For now, only
/// RTLModule instances create a new entity.
static LogicalResult exportTcl(Entity &entity, Operation *op) {
  // Instances require a new child entity and trigger a descent of the
  // instantiated module in the new entity.
  StringRef instName;
  if (auto inst = dyn_cast<rtl::InstanceOp>(op)) {
    instName = inst.instanceName();
    Optional<Entity> inModule = entity.enter(inst);
    if (inModule && failed(exportTcl(*inModule, inst.getReferencedModule())))
      return failure();
  }

  // Iterate through 'op's attributes, looking for attributes which we
  // recognize.
  for (NamedAttribute attr : op->getAttrs()) {
    if (entity.emittedAttrKeys.find(attr.first) != entity.emittedAttrKeys.end())
      op->emitWarning("Attribute has already been emitted: '")
          << attr.first << "'";

    if (auto loc = attr.second.dyn_cast<PhysLocationAttr>())
      if (failed(entity.emit(op, attr.first, instName, loc)))
        return failure();
  }

  auto result = op->walk([&](Operation *innerOp) {
    if (innerOp != op && failed(exportTcl(entity, innerOp)))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

/// Write out all the relevant tcl commands. Create one 'proc' per module (since
/// we don't know which one will be the 'top' module). Said procedure takes the
/// parent entity name since we don't assume that the created module is the top
/// level for the entire design.
LogicalResult circt::msft::exportQuartusTcl(ModuleOp module,
                                            llvm::raw_ostream &os) {
  TclOutputState state(os);

  for (auto &op : module.getBody()->getOperations()) {
    auto rtlMod = dyn_cast<RTLModuleOp>(op);
    if (!rtlMod)
      continue;
    os << "proc " << rtlMod.getName() << "_config { parent } {\n";
    Entity entity(state);
    if (failed(exportTcl(entity, rtlMod)))
      return failure();
    os << "}\n\n";
  }
  return success();
}

void circt::msft::registerMSFTTclTranslation() {
  mlir::TranslateFromMLIRRegistration toQuartusTcl(
      "export-quartus-tcl", exportQuartusTcl,
      [](mlir::DialectRegistry &registry) {
        registry.insert<MSFTDialect, RTLDialect>();
      });
}
