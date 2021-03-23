//===- ExportQuartusTcl.cpp - Emit Quartus-flavored TCL -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Write out TCL with the appropriate API calls for Quartus.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Translation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace rtl;
using namespace msft;

// TODO: Currently assumes Stratix 10 and QuartusPro. Make more general.

namespace {
struct TclOutputState {
  TclOutputState(llvm::raw_ostream &os) : os(os) {}

  llvm::raw_ostream &os;
  llvm::raw_ostream &indent() {
    os.indent(2);
    return os;
  };

  SmallPtrSet<Operation *, 32> modulesEmitted;
};

struct Entity {
  Entity(TclOutputState &s)
      : s(s), parent(nullptr), name("$parent"), insideEmittedModule(false) {}
  Entity(Entity *parent, StringRef name, bool insideEmittedModule)
      : s(parent->s), parent(parent), name(name),
        insideEmittedModule(insideEmittedModule) {}

  Optional<Entity> enter(InstanceOp inst);
  LogicalResult emit(Operation *, StringRef, PhysLocationAttr);

  void emitPath();

  TclOutputState &s;
  Entity *parent;
  StringRef name;
  bool insideEmittedModule;
};
} // anonymous namespace

Optional<Entity> Entity::enter(InstanceOp inst) {
  auto mod = dyn_cast_or_null<rtl::RTLModuleOp>(inst.getReferencedModule());
  if (!mod)
    return {};
  bool modEmitted = insideEmittedModule ||
                    s.modulesEmitted.find(mod) != s.modulesEmitted.end();
  s.modulesEmitted.insert(mod);

  return Entity(this, inst.instanceName(), modEmitted);
}

LogicalResult Entity::emit(Operation *op, StringRef attrKey,
                           PhysLocationAttr pla) {
  if (!attrKey.startswith_lower("loc:")) {
    op->emitError("Error in '")
        << attrKey << "' PhysLocation attribute. Expected loc:<entityName>.";
    return failure();
  }
  StringRef childEntity = attrKey.substr(4);
  s.indent() << "set_location_assignment ";

  char numCharacter;
  switch (pla.getDevType().getValue()) {
  case DeviceType::M20K:
    s.os << "M20K";
    numCharacter = 'N';
    break;
  case DeviceType::DSP:
    s.os << "MSDSP";
    numCharacter = 'N';
    break;
  }

  s.os << "_X" << pla.getX() << "_Y" << pla.getY() << "_" << numCharacter
       << pla.getNum();
  s.os << " -to ";
  emitPath();
  s.os << childEntity << '\n';

  return success();
}

void Entity::emitPath() {
  if (parent)
    parent->emitPath();
  s.os << name << "|";
}

static LogicalResult exportTcl(Entity entity, Operation *op) {
  if (auto inst = dyn_cast<rtl::InstanceOp>(op)) {
    Optional<Entity> inModule = entity.enter(inst);
    if (inModule && failed(exportTcl(*inModule, inst.getReferencedModule())))
      return failure();
  }

  for (NamedAttribute attr : op->getAttrs())
    if (auto loc = attr.second.dyn_cast<PhysLocationAttr>())
      if (failed(entity.emit(op, attr.first, loc)))
        return failure();

  auto result = op->walk([&](Operation *innerOp) {
    if (innerOp != op && failed(exportTcl(entity, innerOp)))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult circt::msft::exportQuartusTCL(ModuleOp module,
                                            llvm::raw_ostream &os) {
  TclOutputState state(os);

  for (auto &op : module.getBody()->getOperations()) {
    auto rtlMod = dyn_cast<RTLModuleOp>(op);
    if (!rtlMod)
      continue;
    os << "proc " << rtlMod.getName() << "_config { parent } {\n";
    if (failed(exportTcl(Entity(state), rtlMod)))
      return failure();
    os << "}\n\n";
  }
  return success();
}

void circt::msft::registerMSFTTclTranslation() {
  mlir::TranslateFromMLIRRegistration toQuartusTcl(
      "export-quartus-tcl", exportQuartusTCL,
      [](mlir::DialectRegistry &registry) {
        registry.insert<MSFTDialect, RTLDialect>();
      });
}
