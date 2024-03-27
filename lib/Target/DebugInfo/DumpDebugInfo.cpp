//===- DumpDebugInfo.cpp - Human-readable debug info dump -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/DebugInfo.h"
#include "circt/Target/DebugInfo.h"
#include "mlir/Support/IndentedOstream.h"

using namespace mlir;
using namespace circt;

static void dump(DIModule &module, raw_indented_ostream &os);

static void dump(DIVariable &variable, raw_indented_ostream &os) {
  os << "Variable " << variable.name;
  if (variable.loc)
    os << " at " << variable.loc;
  os << "\n";
  os.indent();
  if (variable.value) {
    if (auto blockArg = dyn_cast_or_null<BlockArgument>(variable.value)) {
      os << "Arg " << blockArg.getArgNumber() << " of "
         << blockArg.getOwner()->getParentOp()->getName();
    } else if (auto result = dyn_cast_or_null<OpResult>(variable.value)) {
      os << "Result " << result.getResultNumber() << " of "
         << result.getDefiningOp()->getName();
    }
    os << " of type " << variable.value.getType() << " at "
       << variable.value.getLoc() << "\n";
  }
  os.unindent();
}

static void dump(DIInstance &instance, raw_indented_ostream &os) {
  os << "Instance " << instance.name << " of " << instance.module->name;
  if (instance.op)
    os << " for " << instance.op->getName() << " at " << instance.op->getLoc();
  os << "\n";
  if (instance.module->isInline) {
    os.indent();
    dump(*instance.module, os);
    os.unindent();
  }
}

static void dump(DIModule &module, raw_indented_ostream &os) {
  os << "Module " << module.name;
  if (module.op)
    os << " for " << module.op->getName() << " at " << module.op->getLoc();
  os << "\n";
  os.indent();
  for (auto *variable : module.variables)
    dump(*variable, os);
  for (auto *instance : module.instances)
    dump(*instance, os);
  os.unindent();
}

static void dump(DebugInfo &di, raw_indented_ostream &os) {
  os << "DebugInfo for " << di.operation->getName() << " at "
     << di.operation->getLoc() << "\n";
  os.indent();
  for (auto nameAndModule : di.moduleNodes)
    dump(*nameAndModule.second, os);
  os.unindent();
}

LogicalResult debug::dumpDebugInfo(Operation *module, llvm::raw_ostream &os) {
  DebugInfo di(module);
  raw_indented_ostream indentedOs(os);
  dump(di, indentedOs);
  return success();
}
