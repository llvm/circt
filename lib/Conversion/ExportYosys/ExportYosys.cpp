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

struct RTLILConverter {};

struct ExprEmitter
    : public hw::TypeOpVisitor<ExprEmitter, FailureOr<Yosys::SigSpec>>,
      public comb::CombinationalVisitor<ExprEmitter,
                                        FailureOr<Yosys::SigSpec>> {
  using hw::TypeOpVisitor<ExprEmitter, ResultTy>::visitTypeOp;
  using comb::CombinationalVisitor<ExprEmitter, ResultTy>::visitComb;
  FailureOr<Yosys::SigSpec> emitExpression(Operation* op) {}
}

using ResultTy = LogicalResult;
struct ModuleConverter
    : public hw::TypeOpVisitor<ModuleConverter, ResultTy>,
      public comb::CombinationalVisitor<ModuleConverter, ResultTy> {
  using hw::TypeOpVisitor<ModuleConverter, ResultTy>::visitTypeOp;
  using comb::CombinationalVisitor<ModuleConverter, ResultTy>::visitComb;
  FailureOr<SigSpec> getValue(Value value) {
    auto it = mapping.find(value);
    if (it != mapping.end())
      return it->second;
    if (isa<OpResult>(value)) {
      // TODO: If it's instance like ...
      auto *op = value.getDefiningOp();
      auto result = ExprEmitter().emitExpression(op);
      if (failed(result))
        return result;

      setLowering(op->getResult(0), result.value());
      return success();
    }
    // TODO: Convert ports.
    return failure();
  }

  ResultTy visitTypeOp(ConstantOp op) {
    if (op.getValue().getBitWidth() >= 32)
      return op.emitError() << "unsupported";
    // TODO: Figure out who has to free constants and sigspec.
    return allocateSigSpec(new RTLIL::Const(op.getValue().getZExtValue()));
  }

  ResultTy visit(Operation *op) { return dispatchCombinationalVisitor(op); }
  ResultTy visitInvalidComb(Operation *op) { return dispatchTypeOpVisitor(op); }
  ResultTy visitUnhandledComb(Operation *op) { return visitUnhandledExpr(op); }
  ResultTy visitInvalidTypeOp(Operation *op) { return visitUnhandledExpr(op); }
  ResultTy visitUnhandledExpr(Operation *op) {
    return op->emitError() << " is unsupported";
  }

  // TODO: Consider BumpAllocator instead of many allocations with unique
  // pointers.
  llvm::SmallVector<std::unique_ptr<SigSpec>> sigSpecs;
  template <typename... Params> SigSpec allocateSigSpec(Params &&...params) {
    return SigSpec(std::forward<Params>(params)...));
  }

  // Comb.
  // ResultTy visitComb(MuxOp op);
  template <typename BinaryFn>
  ResultTy emitVariadicOp(Operation *op, BinaryFn fn) {
    // Construct n-1 binary op (currently linear) chains.
    // TODO: Need to create a tree?
    std::optional<SigSpec> cur;
    for (auto operand : op->getOperands()) {
      auto result = getValue(operand);
      if (failed(result))
        return result;
      if (cur) {
        auto *resultWire = rtlilModule->addWire(
            getNewName(), hw::getBitWidth(operand.getType()));

        Cell *cell = fn(getNewName(), *cur, result.value(), resultWire);
        // assert(cell->connections().size() == 3);
        cur = allocateSigSpec(resultWire);
      } else
        cur = result.value();
    }
    return cur.value();
  }
  circt::Namespace moduleNameSpace;
  RTLIL::IdString getNewName(StringRef name = "_GEN_") {
    return getEscapedName(moduleNameSpace.newName(name));
  }

  ResultTy visitComb(AddOp op) {
    return emitVariadicOp(op, [&](auto name, auto l, auto r, auto out) {
      return rtlilModule->addAdd(name, l, r, out);
    });
  }
  ResultTy visitComb(SubOp op) {
    return emitVariadicOp(op, [&](auto name, auto l, auto r, auto out) {
      return rtlilModule->addSub(name, l, r, out);
    });
  }

  LogicalResult visitStmt(OutputOp op) {
    assert(op.getNumOperands() == outputs.size());
    for (auto [wire, op] : llvm::zip(outputs, op.getOperands())) {
      auto result = getValue(op);
      if (failed(result))
        return result;
      rtlilModule->connect(*allocateSigSpec(wire), *result.value());
    }
    return success();
  }

  ResultTy visitComb(MulOp op) {}
  ResultTy visitComb(DivUOp op) {}
  ResultTy visitComb(DivSOp op) {}
  ResultTy visitComb(ModUOp op) {}
  ResultTy visitComb(ModSOp op) {}
  ResultTy visitComb(ShlOp op) {}
  ResultTy visitComb(ShrUOp op) {}
  ResultTy visitComb(ShrSOp op) {}
  ResultTy visitComb(AndOp op) {
    return emitVariadicOp(op, [&](auto name, auto l, auto r, auto out) {
      return rtlilModule->addAnd(name, l, r, out);
    });
  }
  ResultTy visitComb(OrOp op) {
    return emitVariadicOp(op, [&](auto name, auto l, auto r, auto out) {
      return rtlilModule->addOr(name, l, r, out);
    });
  }
  ResultTy visitComb(XorOp op) {}

  // SystemVerilog spec 11.8.1: "Reduction operator results are unsigned,
  // regardless of the operands."
  ResultTy visitComb(ParityOp op) {}

  // ResultTy visitComb(ReplicateOp op);
  // ResultTy visitComb(ConcatOp op);
  // ResultTy visitComb(ExtractOp op);
  // ResultTy visitComb(ICmpOp op);

  LogicalResult run() {
    return LogicalResult::failure(
        module
            .walk<mlir::WalkOrder::PreOrder>([this](Operation *op) {
              // Skip zero result operatins.
              op->emitWarning() << "foo";
              RTLIL_BACKEND::dump_design(std::cout, design, false);
              if (module == op)
                return WalkResult::advance();
              if (auto out = dyn_cast<hw::OutputOp>(op))
                return visitStmt(out), WalkResult::advance();
              if (failed(visit(op)))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted());
  }

  ModuleConverter(Yosys::RTLIL::Design *design,
                  Yosys::RTLIL::Module *rtlilModule, hw::HWModuleOp module)
      : design(design), rtlilModule(rtlilModule), module(module) {
    ModulePortInfo ports(module.getPortList());
    size_t inputPos = 0;
    for (auto [idx, port] : llvm::enumerate(ports)) {
      auto *wire = rtlilModule->addWire(getEscapedName(port.getName()));
      wire->port_id = idx;
      auto bitWidth = hw::getBitWidth(port.type);
      assert(bitWidth >= 0);
      wire->width = bitWidth;
      if (port.isOutput()) {
        wire->port_output = true;
        outputs.push_back(wire);
      } else {
        setLowering(module.getBodyBlock()->getArgument(inputPos++),
                    allocateSigSpec(wire));
        // TODO: Need to consider inout?
        wire->port_input = true;
      }
    }
  }

  DenseMap<Value, SigSpec> mapping;
  SmallVector<RTLIL::Wire *> outputs;

  LogicalResult setLowering(Value value, SigSpec s) {
    return LogicalResult::success(mapping.insert({value, s}).second);
  }

  Yosys::RTLIL::Design *design;
  Yosys::RTLIL::Module *rtlilModule;
  hw::HWModuleOp module;
  RTLIL::Wire *getWireForValue(Value value);
  RTLIL::Cell *getCellForValue(Value value);
};

} // namespace

void ExportYosysPass::runOnOperation() {
  auto *design = new Yosys::RTLIL::Design;
  // auto *mod = new Yosys::RTLIL::Module;
  // mod->name = "\\dog";
  // design->add(mod);
  // auto *wire = mod->addWire("\\cat");
  // mod->addCell()
  // wire->port_id = 1;
  // wire->port_input = true;
  // design->addModule(Yosys::RTLIL::IdString("test"));
  SmallVector<ModuleConverter> converter;
  for (auto op : getOperation().getOps<hw::HWModuleOp>()) {
    llvm::errs() << op.getModuleName() << " "
                 << getEscapedName(op.getModuleName());
    auto *newModule = design->addModule(getEscapedName(op.getModuleName()));
    converter.emplace_back(design, newModule, op);
  }
  for (auto &c : converter)
    c.run();

  RTLIL_BACKEND::dump_design(std::cout, design, false);
  Yosys::run_pass("opt", design);
  RTLIL_BACKEND::dump_design(std::cout, design, false);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> circt::createExportYosys() {
  return std::make_unique<ExportYosysPass>();
}
