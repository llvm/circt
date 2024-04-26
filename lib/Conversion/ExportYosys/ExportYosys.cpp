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
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/Seq/SeqVisitor.h"
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

struct YosysCircuitImporter;

struct ModuleConverter
    : public hw::TypeOpVisitor<ModuleConverter, LogicalResult>,
      public hw::StmtVisitor<ModuleConverter, LogicalResult>,
      public comb::CombinationalVisitor<ModuleConverter, LogicalResult>,
      public seq::SeqOpVisitor<ModuleConverter, LogicalResult> {

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
    return setValue(value, wire);
  }

  LogicalResult setValue(Value value, RTLIL::Wire *wire) {
    return success(mapping.insert({value, wire}).second);
  }

  FailureOr<SigSpec> getValue(Value value) {
    auto it = mapping.find(value);
    if (it != mapping.end())
      return SigSpec(it->second);
    return failure();
  }

  RTLIL::Wire *getValueWire(Value value) {
    auto it = mapping.find(value);
    if (it != mapping.end())
      return it->second;
    return nullptr;
  }

  circt::Namespace moduleNameSpace;
  RTLIL::IdString getNewName(StringRef name = "") {
    if (name.empty())
      return NEW_ID;

    return getEscapedName(moduleNameSpace.newName(name));
  }

  LogicalResult lowerPorts();
  LogicalResult lowerBody();

  ModuleConverter(YosysCircuitImporter &circuitConverter,
                  Yosys::RTLIL::Module *rtlilModule, hw::HWModuleOp module)
      : circuitConverter(circuitConverter), rtlilModule(rtlilModule),
        module(module) {}

  YosysCircuitImporter &circuitConverter;

  DenseMap<Value, RTLIL::Wire *> mapping;
  SmallVector<RTLIL::Wire *> outputs;

  LogicalResult setLowering(Value value, SigSpec s) {
    auto it = mapping.find(value);
    assert(it != mapping.end());
    rtlilModule->connect(it->second, s);
    return success();
  }

  template <typename SigSpecTy>
  LogicalResult setLowering(Value value, SigSpecTy s) {
    return setLowering(value, SigSpec(s));
  }

  Yosys::RTLIL::Module *rtlilModule;
  hw::HWModuleOp module;
  using hw::TypeOpVisitor<ModuleConverter, LogicalResult>::visitTypeOp;
  using hw::StmtVisitor<ModuleConverter, LogicalResult>::visitStmt;
  using comb::CombinationalVisitor<ModuleConverter, LogicalResult>::visitComb;
  using seq::SeqOpVisitor<ModuleConverter, LogicalResult>::visitSeq;

  LogicalResult visitOp(Operation *op) {
    return dispatchCombinationalVisitor(op);
  }
  LogicalResult visitUnhandledTypeOp(Operation *op) {
    return op->emitError() << "unsupported op";
  }
  LogicalResult visitUnhandledExpr(Operation *op) {
    return op->emitError() << "unsupported op";
  }
  LogicalResult visitInvalidComb(Operation *op) {
    return dispatchTypeOpVisitor(op);
  }
  LogicalResult visitUnhandledComb(Operation *op) {
    return visitUnhandledExpr(op);
  }
  LogicalResult visitInvalidTypeOp(Operation *op) {
    return dispatchStmtVisitor(op);
  }
  LogicalResult visitInvalidStmt(Operation *op) {
    return dispatchSeqOpVisitor(op);
  }
  LogicalResult visitUnhandledStmt(Operation *op) {
    return visitUnhandledExpr(op);
  }
  LogicalResult visitInvalidSeqOp(Operation *op) {
    return visitUnhandledExpr(op);
  }
  LogicalResult visitUnhandledSeqOp(Operation *op) {
    return visitUnhandledExpr(op);
  }

  // HW type op.
  LogicalResult visitTypeOp(ConstantOp op) {
    if (op.getValue().getBitWidth() >= 33) {
      // FIXME: Don't use inefficient binary encoding.
      std::vector<bool> result(op.getValue().getBitWidth(), false);
      for (size_t i = 0; i < op.getValue().getBitWidth(); i++) {
        result[i] = op.getValue()[i];
      }

      return setLowering(op, RTLIL::Const(result));
    }

    return setLowering(op, RTLIL::Const(op.getValue().getZExtValue(),
                                        op.getValue().getBitWidth()));
  }

  // HW stmt op.
  LogicalResult visitStmt(OutputOp op);
  LogicalResult visitStmt(InstanceOp op);

  // HW expr ops.
  LogicalResult visitTypeOp(AggregateConstantOp op);
  LogicalResult visitTypeOp(ArrayCreateOp op);
  LogicalResult visitTypeOp(ArrayGetOp op);

  // Comb op.
  LogicalResult visitComb(AddOp op);
  LogicalResult visitComb(SubOp op);
  LogicalResult visitComb(MulOp op);
  LogicalResult visitComb(AndOp op);
  LogicalResult visitComb(OrOp op);
  LogicalResult visitComb(XorOp op);
  LogicalResult visitComb(MuxOp op);
  LogicalResult visitComb(ExtractOp op);
  LogicalResult visitComb(ICmpOp op);
  LogicalResult visitComb(ConcatOp op);
  LogicalResult visitComb(ShlOp op);
  LogicalResult visitComb(ShrSOp op);
  LogicalResult visitComb(ShrUOp op);
  LogicalResult visitComb(ReplicateOp op);

  // Seq op.
  LogicalResult visitSeq(seq::FirRegOp op);
  LogicalResult visitSeq(seq::FirMemOp op);
  LogicalResult visitSeq(seq::FirMemWriteOp op);
  LogicalResult visitSeq(seq::FirMemReadOp op);
  LogicalResult visitSeq(seq::FirMemReadWriteOp op);

  template <typename BinaryFn>
  LogicalResult emitVariadicOp(Operation *op, BinaryFn fn) {
    // Construct n-1 binary op (currently linear) chains.
    // TODO: Need to create a tree?
    std::optional<SigSpec> cur;
    for (auto operand : op->getOperands()) {
      auto result = getValue(operand);
      if (failed(result))
        return failure();
      cur = cur ? fn(getNewName(), *cur, result.value()) : result.value();
    }

    return setLowering(op->getResult(0), cur.value());
  }

  template <typename BinaryFn>
  LogicalResult emitBinaryOp(Operation *op, BinaryFn fn) {
    assert(op->getNumOperands() == 2 && "only expect binary op");
    auto lhs = getValue(op->getOperand(0));
    auto rhs = getValue(op->getOperand(1));
    if (failed(lhs) || failed(rhs))
      return failure();
    return setLowering(op->getResult(0),
                       fn(getNewName(), lhs.value(), rhs.value()));
  }
};
struct YosysCircuitImporter {
  YosysCircuitImporter(mlir::ModuleOp module, Yosys::RTLIL::Design *design,
                       InstanceGraph *instanceGraph)
      : module(module), design(design), instanceGraph(instanceGraph) {}
  llvm::DenseMap<StringAttr, Yosys::RTLIL::Module *> moduleMapping;
  LogicalResult run() {
    SmallVector<ModuleConverter> converter;
    for (auto op : module.getOps<hw::HWModuleOp>()) {
      auto *newModule = design->addModule(getEscapedName(op.getModuleName()));
      moduleMapping[op.getModuleNameAttr()] = newModule;
      converter.emplace_back(*this, newModule, op);
      if (failed(converter.back().lowerPorts()))
        return failure();
    }
    for (auto &c : converter)
      if (failed(c.lowerBody()))
        return failure();
    return success();
  }
  mlir::ModuleOp module;
  Yosys::RTLIL::Design *design;
  InstanceGraph *instanceGraph;
};
} // namespace

LogicalResult ModuleConverter::lowerPorts() {
  ModulePortInfo ports(module.getPortList());
  size_t inputPos = 0;
  for (auto [idx, port] : llvm::enumerate(ports)) {
    auto *wire = createWire(port.type, port.name);
    if (!wire)
      return mlir::emitError(port.loc) << "unknown type";
    // NOTE: Port id is 1-indexed.
    wire->port_id = idx + 1;
    if (port.isOutput()) {
      wire->port_output = true;
      outputs.push_back(wire);
    } else if (port.isInput()) {
      setValue(module.getBodyBlock()->getArgument(inputPos++), wire);
      // TODO: Need to consider inout?
      wire->port_input = true;
    } else {
      return module.emitError() << "inout is unssuported";
    }
  }
  // Need to call fixup ports after port mutations.
  rtlilModule->fixup_ports();
  return success();
}

LogicalResult ModuleConverter::lowerBody() {
  module.walk([this](Operation *op) {
    for (auto result : op->getResults()) {
      // TODO: Use SSA name.
      mlir::StringAttr name = {};
      createAndSetWire(result, name);
    }
  });

  auto result =
      module
          .walk<mlir::WalkOrder::PostOrder>([this](Operation *op) {
            if (module == op)
              return WalkResult::advance();

            if (isa<comb::CombDialect, hw::HWDialect, seq::SeqDialect>(
                    op->getDialect())) {
              LLVM_DEBUG(llvm::dbgs() << "Visiting " << *op << "\n");
              if (failed(visitOp(op)))
                return WalkResult::interrupt();
              LLVM_DEBUG(llvm::dbgs() << "Success \n");
            } else {
              // Ignore Verif, LTL, SV and so on.
            }
            return WalkResult::advance();
          })
          .wasInterrupted();
  return LogicalResult::success(!result);
}

//===----------------------------------------------------------------------===//
// HW Ops.
//===----------------------------------------------------------------------===//

LogicalResult ModuleConverter::visitStmt(OutputOp op) {
  assert(op.getNumOperands() == outputs.size());
  for (auto [wire, op] : llvm::zip(outputs, op.getOperands())) {
    auto result = getValue(op);
    if (failed(result))
      return failure();
    rtlilModule->connect(Yosys::SigSpec(wire), result.value());
  }

  return success();
}

LogicalResult ModuleConverter::visitStmt(InstanceOp op) {
  auto it =
      circuitConverter.moduleMapping.find(op.getModuleNameAttr().getAttr());
  IdString id;
  if (it == circuitConverter.moduleMapping.end()) {
    // Ext module.
    auto referredMod =
        circuitConverter.instanceGraph->lookup(op.getModuleNameAttr().getAttr())
            ->getModule();
    if (auto extmod =
            dyn_cast<hw::HWModuleExternOp>(referredMod.getOperation()))
      id = getEscapedName(extmod.getVerilogModuleName());
  } else {
    id = it->second->name;
  }

  auto *cell = rtlilModule->addCell(getEscapedName(op.getInstanceName()), id);
  auto connect = [&](ArrayAttr names, auto values) -> LogicalResult {
    for (auto [portName, value] :
         llvm::zip(names.getAsRange<StringAttr>(), values)) {
      auto loweredValue = getValue(value);
      if (failed(loweredValue))
        return failure();
      cell->connections_.insert(
          {getEscapedName(portName), loweredValue.value()});
    }
    return success();
  };

  if (failed(connect(op.getArgNames(), op.getOperands())) ||
      failed(connect(op.getResultNames(), op.getResults())))
    return failure();

  return success();
}

LogicalResult ModuleConverter::visitTypeOp(AggregateConstantOp op) {
  return failure();
}

LogicalResult ModuleConverter::visitTypeOp(ArrayCreateOp op) {
  return failure();
}

LogicalResult ModuleConverter::visitTypeOp(ArrayGetOp op) { return failure(); }

//===----------------------------------------------------------------------===//
// Comb Ops.
//===----------------------------------------------------------------------===//

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

LogicalResult ModuleConverter::visitComb(AndOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->And(name, l, r);
  });
}

LogicalResult ModuleConverter::visitComb(MulOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Mul(name, l, r);
  });
}

LogicalResult ModuleConverter::visitComb(OrOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Or(name, l, r);
  });
}

LogicalResult ModuleConverter::visitComb(XorOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Xor(name, l, r);
  });
}

LogicalResult ModuleConverter::visitComb(ExtractOp op) {
  auto result = getValue(op.getOperand());
  if (failed(result))
    return result;
  auto sig = result.value().extract(op.getLowBit(),
                                    op.getType().getIntOrFloatBitWidth());
  return setLowering(op, sig);
}

LogicalResult ModuleConverter::visitComb(ConcatOp op) {
  SigSpec ret;
  for (auto operand : op.getOperands()) {
    auto result = getValue(operand);
    if (failed(result))
      return result;
    ret.append(result.value());
  }
  // TODO: Check endian.
  return setLowering(op, ret);
}

LogicalResult ModuleConverter::visitComb(ShlOp op) {
  return emitBinaryOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Shl(name, l, r);
  });
}

LogicalResult ModuleConverter::visitComb(ShrUOp op) {
  return emitBinaryOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Shr(name, l, r);
  });
}
LogicalResult ModuleConverter::visitComb(ShrSOp op) {
  return emitBinaryOp(op, [&](auto name, auto l, auto r) {
    // TODO: Make sure it's correct
    return rtlilModule->Sshr(name, l, r);
  });
}

LogicalResult ModuleConverter::visitComb(ReplicateOp op) {
  auto value = getValue(op.getOperand());
  if (failed(value))
    return failure();
  return setLowering(op, value.value().repeat(op.getMultiple()));
}

LogicalResult ModuleConverter::visitComb(ICmpOp op) {
  return emitBinaryOp(op, [&](auto name, auto l, auto r) {
    switch (op.getPredicate()) {
    case ICmpPredicate::eq:
    case ICmpPredicate::ceq:
    case ICmpPredicate::weq:
      return rtlilModule->Eq(name, l, r);
    case ICmpPredicate::ne:
    case ICmpPredicate::cne:
    case ICmpPredicate::wne:
      return rtlilModule->Ne(name, l, r);
    case ICmpPredicate::slt:
      return rtlilModule->Lt(name, l, r, /*is_signed=*/true);
    case ICmpPredicate::sle:
      return rtlilModule->Le(name, l, r, /*is_signed=*/true);
    case ICmpPredicate::sgt:
      return rtlilModule->Gt(name, l, r, /*is_signed=*/true);
    case ICmpPredicate::sge:
      return rtlilModule->Ge(name, l, r, /*is_signed=*/true);
    case ICmpPredicate::ult:
      return rtlilModule->Lt(name, l, r, /*is_signed=*/false);
    case ICmpPredicate::ule:
      return rtlilModule->Le(name, l, r, /*is_signed=*/false);
    case ICmpPredicate::ugt:
      return rtlilModule->Gt(name, l, r, /*is_signed=*/false);
    case ICmpPredicate::uge:
      return rtlilModule->Ge(name, l, r, /*is_signed=*/false);
    default:
      llvm::report_fatal_error("unsupported icmp predicate");
    }
  });
}

//===----------------------------------------------------------------------===//
// Seq Ops.
//===----------------------------------------------------------------------===//

LogicalResult ModuleConverter::visitSeq(seq::FirRegOp op) {
  auto result = getValue(op.getResult());
  auto clock = getValue(op.getClk());
  auto next = getValue(op.getNext());

  if (failed(result) || failed(clock) || failed(next))
    return failure();
  if (op.getIsAsync())
    // Adlatch.
    return op.emitError() << "async not supported yet";

  if (op.getReset()) {
    // addSdff
    auto reset = getValue(op.getReset());
    if (failed(reset))
      return reset;
    rtlilModule->addSdff(
        getNewName(op.getName()), clock.value(), reset.value(), next.value(),
        result.value(),
        /*FIXME*/ RTLIL::Const(0, getBitWidthSeq(op.getType())));
    return success();
  }

  rtlilModule->addDff(getNewName(op.getName()), clock.value(), next.value(),
                      result.value());
  return success();
}

LogicalResult ModuleConverter::visitSeq(seq::FirMemOp op) { return failure(); }

LogicalResult ModuleConverter::visitSeq(seq::FirMemWriteOp op) {
  return failure();
}
LogicalResult ModuleConverter::visitSeq(seq::FirMemReadOp op) {
  return failure();
}
LogicalResult ModuleConverter::visitSeq(seq::FirMemReadWriteOp op) {
  return failure();
}

void ExportYosysPass::runOnOperation() {
  // Set up yosys.
  Yosys::log_streams.push_back(&std::cerr);
  Yosys::log_error_stderr = true;
  Yosys::yosys_setup();
  auto theDesign = std::make_unique<Yosys::RTLIL::Design>();
  auto *design = theDesign.get();
  auto &theInstanceGraph = getAnalysis<hw::InstanceGraph>();
  YosysCircuitImporter exporter(getOperation(), design, &theInstanceGraph);
  if (failed(exporter.run()))
    return signalPassFailure();

  // RTLIL_BACKEND::dump_design(std::cout, design, false);
  // Yosys::run_pass("hierarchy -top DigitalTop", design);
  Yosys::run_pass("synth", design);
  Yosys::run_pass("write_verilog synth.v", design);
  // RTLIL_BACKEND::dump_design(std::cout, design, false);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> circt::createExportYosys() {
  return std::make_unique<ExportYosysPass>();
}
