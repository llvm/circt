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
#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Debug.h"

#include "kernel/rtlil.h"
#include "kernel/yosys.h"

#define DEBUG_TYPE "export-yosys"

using namespace circt;
using namespace hw;
using namespace comb;
using namespace Yosys;

namespace {
#define GEN_PASS_DEF_EXPORTYOSYS
#include "circt/Conversion/Passes.h.inc"

struct ExportYosysPass : public impl::ExportYosysBase<ExportYosysPass> {
  void runOnOperation() override;
};

std::string getEscapedName(StringRef name) {
  return RTLIL::escape_id(name.str());
}

int64_t getBitWidthSeq(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;
  return getBitWidth(type);
}

struct ModuleConverter
    : public hw::TypeOpVisitor<ModuleConverter, LogicalResult>,
      public hw::StmtVisitor<ModuleConverter, LogicalResult>,
      public comb::CombinationalVisitor<ModuleConverter, LogicalResult> {

  Yosys::Wire *createWire(Type type, StringAttr name) {
    int64_t width = getBitWidthSeq(type);

    if (width < 0)
      return nullptr;

    auto *wire =
        rtlilModule->addWire(name ? getNewName(name) : getNewName(), width);
    return wire;
  }

  LogicalResult createAndSetWire(Value value, StringAttr name) {
    auto *wire = createWire(value.getType(), name);
    if (!wire)
      return failure();
    return setValue(value, SigSpec(wire));
  }

  LogicalResult setValue(Value value, Yosys::SigSpec sig) {
    return success(mapping.insert({value, sig}).second);
  }

  FailureOr<SigSpec> getValue(Value value) {
    auto it = mapping.find(value);
    if (it != mapping.end())
      return it->second;
    return failure();
  }

  circt::Namespace moduleNameSpace;
  RTLIL::IdString getNewName(StringRef name = "_GEN_") {
    return getEscapedName(moduleNameSpace.newName(name));
  }

  FailureOr<SigSpec> visitSeq(seq::FirRegOp op) {
    if (op.getReset())
      return failure();
    auto clock = getValue(op.getClk());
    auto next = getValue(op.getNext());
    auto width = hw::getBitWidth(op.getType());
    if (failed(clock) || failed(next) || width <= 0)
      return failure();
    auto wireName = getNewName(op.getName());
    // Connect!
    auto wire = getValue(op).value();
    rtlilModule->addDff(getNewName(op.getName()), clock.value(), next.value(),
                        SigSpec(wire));
    return SigSpec(wire);
  }

  LogicalResult visitStmt(OutputOp op) {
    assert(op.getNumOperands() == outputs.size());
    for (auto [wire, op] : llvm::zip(outputs, op.getOperands())) {
      auto result = getValue(op);
      if (failed(result))
        return failure();
      rtlilModule->connect(Yosys::SigSpec(wire), result.value());
    }

    return success();
  }

  LogicalResult run() {
    ModulePortInfo ports(module.getPortList());
    size_t inputPos = 0;
    for (auto [idx, port] : llvm::enumerate(ports)) {
      auto *wire = createWire(port.type, port.name);
      // NOTE: Port id is 1-indexed.
      wire->port_id = idx + 1;
      if (port.isOutput()) {
        wire->port_output = true;
        outputs.push_back(wire);
      } else if (port.isInput()) {
        setValue(module.getBodyBlock()->getArgument(inputPos++),
                 Yosys::RTLIL::SigSpec(wire));
        // TODO: Need to consider inout?
        wire->port_input = true;
      } else {
        return failure();
      }
    }
    // Need to call fixup ports after port mutations.
    rtlilModule->fixup_ports();
    module.walk([this](Operation *op) {
      for (auto result : op->getResults()) {
        // TODO: Use SSA name.
        mlir::StringAttr name = {};
        createAndSetWire(result, name);
      }
    });

    auto result = module
                      .walk<mlir::WalkOrder::PostOrder>([this](Operation *op) {
                        if (module == op)
                          return WalkResult::advance();

                        // if (auto out = dyn_cast<hw::OutputOp>(op)) {
                        //   auto result = visitStmt(out);
                        //   if (failed(result))
                        //     return op->emitError() << "failed to lower",
                        //            WalkResult::interrupt();
                        //   return WalkResult::advance();
                        // }

                        if (auto reg = dyn_cast<seq::FirRegOp>(op))
                          visitSeq(reg);
                        else
                          visitOp(op);
                        return WalkResult::advance();
                      })
                      .wasInterrupted();
    return LogicalResult::success(!result);
  }

  ModuleConverter(Yosys::RTLIL::Design *design,
                  Yosys::RTLIL::Module *rtlilModule, hw::HWModuleOp module)
      : design(design), rtlilModule(rtlilModule), module(module) {}

  DenseMap<Value, SigSpec> mapping;
  SmallVector<RTLIL::Wire *> outputs;

  LogicalResult setLowering(Value value, SigSpec s) {
    auto it = mapping.find(value);
    assert(it != mapping.end());
    rtlilModule->connect(it->second, s);
    return success();
  }

  Yosys::RTLIL::Design *design;
  Yosys::RTLIL::Module *rtlilModule;
  hw::HWModuleOp module;
  RTLIL::Wire *getWireForValue(Value value);
  RTLIL::Cell *getCellForValue(Value value);

  using hw::TypeOpVisitor<ModuleConverter, LogicalResult>::visitTypeOp;
  using hw::StmtVisitor<ModuleConverter, LogicalResult>::visitStmt;
  using comb::CombinationalVisitor<ModuleConverter, LogicalResult>::visitComb;

  LogicalResult visitOp(Operation *op) {
    return dispatchCombinationalVisitor(op);
  }
  LogicalResult visitUnhandledTypeOp(Operation *op) {
    return op->emitError() << " is unsupported";
  }
  LogicalResult visitUnhandledExpr(Operation *op) {
    return op->emitError() << " is unsupported";
  }
  LogicalResult visitInvalidComb(Operation *op) {
    return dispatchTypeOpVisitor(op);
  }
  LogicalResult visitUnhandledComb(Operation *op) {
    return visitUnhandledExpr(op);
  }
  LogicalResult visitInvalidTypeOp(Operation *op) {
    return visitUnhandledExpr(op);
  }

  LogicalResult visitTypeOp(ConstantOp op) {
    if (op.getValue().getBitWidth() >= 32)
      return op.emitError() << "unsupported";
    return setLowering(
        op, Yosys::SigSpec(RTLIL::Const(op.getValue().getZExtValue(),
                                        op.getValue().getBitWidth())));
  }

  LogicalResult visitComb(AddOp op);
  LogicalResult visitComb(SubOp op);
  LogicalResult visitComb(MuxOp op);

  // Comb.
  // ResultTy visitComb(MuxOp op);
  template <typename BinaryFn>
  LogicalResult emitVariadicOp(Operation *op, BinaryFn fn) {
    // Construct n-1 binary op (currently linear) chains.
    // TODO: Need to create a tree?
    std::optional<SigSpec> cur;
    for (auto operand : op->getOperands()) {
      auto result = getValue(operand);
      if (failed(result))
        return failure();
      if (cur) {
        cur = fn(getNewName(), *cur, result.value());
      } else
        cur = result.value();
    }

    return setLowering(op->getResult(0), cur.value());
  }
};
} // namespace

LogicalResult ModuleConverter::visitComb(AddOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Add(name, l, r);
  });
}

LogicalResult ModuleConverter::visitComb(SubOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Sub(name, l, r);
  });
}

LogicalResult ModuleConverter::visitComb(MuxOp op) {
  auto cond = getValue(op.getCond());
  auto high = getValue(op.getTrueValue());
  auto low = getValue(op.getFalseValue());
  if (failed(cond) || failed(high) || failed(low))
    return failure();

  return setLowering(op, rtlilModule->Mux(getNewName(), low.value(),
                                          high.value(), cond.value()));
}

void ExportYosysPass::runOnOperation() {
  // Set up yosys.
  Yosys::log_streams.push_back(&std::cout);
  Yosys::log_error_stderr = true;
  Yosys::yosys_setup();

  auto *design = new Yosys::RTLIL::Design;
  SmallVector<ModuleConverter> converter;
  for (auto op : getOperation().getOps<hw::HWModuleOp>()) {
    auto *newModule = design->addModule(getEscapedName(op.getModuleName()));
    converter.emplace_back(design, newModule, op);
  }
  for (auto &c : converter)
    if (failed(c.run()))
      signalPassFailure();

  RTLIL_BACKEND::dump_design(std::cout, design, false);
  Yosys::run_pass("synth", design);
  Yosys::run_pass("write_verilog synth.v", design);
  // Yosys::shell(design);

  RTLIL_BACKEND::dump_design(std::cout, design, false);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> circt::createExportYosys() {
  return std::make_unique<ExportYosysPass>();
}
