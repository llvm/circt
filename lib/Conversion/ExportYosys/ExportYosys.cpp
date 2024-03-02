//===- LowerFirMem.cpp - Seq FIRRTL memory lowering -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq FirMem ops to instances of HW generated modules.
//
//===----------------------------------------------------------------------===//
#include "circt/Conversion/ExportYosys.h"
#include "../PassDetail.h"
#include "backends/rtlil/rtlil_backend.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Namespace.h"
#include "kernel/rtlil.h"
#include "kernel/yosys.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "export-yosys"

using namespace circt;
using namespace hw;
using namespace Yosys;

namespace {
#define GEN_PASS_DEF_EXPORTYOSYS
#include "circt/Conversion/Passes.h.inc"

struct ExportYosysPass : public impl::ExportYosysBase<ExportYosysPass> {
  void runOnOperation() override;
};

const std::string &getEscapedName(StringRef name) {
  return RTLIL::escape_id(name.str());
}

struct RTLILConverter {};

struct ExprEmitter : public TypeOpVisitor<ExprEmitter, RTLIL::Wire *>,
                     public CombinationalVisitor<ExprEmitter, RTLIL::Wire *> {};

struct ModuleConverter {
  ModuleConverter(Yosys::RTLIL::Design *design,
                  Yosys::RTLIL::Module *rtlilModule, hw::HWModuleOp module)
      : design(design), rtlilModule(rtlilModule), module(module) {}

  Yosys::RTLIL::Design *design;
  Yosys::RTLIL::Module *rtlilModule;
  hw::HWModuleOp module;
  RTLIL::Wire *getWireForValue(Value value);
  RTLIL::Cell *getCellForValue(Value value);
};

} // namespace

void ExportYosysPass::runOnOperation() {
  auto *design = new Yosys::RTLIL::Design;
  auto *mod = new Yosys::RTLIL::Module;
  mod->name = "\\dog";
  design->add(mod);
  auto *wire = mod->addWire("\\cat");
  // wire->port_id = 1;
  // wire->port_input = true;
  // design->addModule(Yosys::RTLIL::IdString("test"));
  RTLIL_BACKEND::dump_design(std::cout, design, false);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> circt::createExportYosys() {
  return std::make_unique<ExportYosysPass>();
}
