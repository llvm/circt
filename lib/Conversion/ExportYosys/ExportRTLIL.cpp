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
#include "circt/Conversion/ExportRTLIL.h"
#include "../PassDetail.h"
#include "RTLILConverterInternal.h"
#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqVisitor.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"

// Yosys headers.
#include "backends/rtlil/rtlil_backend.h"
#include "kernel/rtlil.h"
#include "kernel/yosys.h"

#define DEBUG_TYPE "export-yosys"

using namespace circt;
using namespace hw;
using namespace comb;
using namespace Yosys;
using namespace rtlil;

std::string circt::rtlil::getEscapedName(StringRef name) {
  return RTLIL::escape_id(name.str());
}

namespace {
int64_t getBitWidthSeq(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;
  return getBitWidth(type);
}

struct ExportRTLILDesign;
struct ExportRTLILModule
    : public hw::TypeOpVisitor<ExportRTLILModule, LogicalResult>,
      public hw::StmtVisitor<ExportRTLILModule, LogicalResult>,
      public comb::CombinationalVisitor<ExportRTLILModule, LogicalResult>,
      public seq::SeqOpVisitor<ExportRTLILModule, LogicalResult> {

  FailureOr<RTLIL::Cell *>
  createCell(llvm::StringRef cellName, llvm::StringRef instanceName,
             ArrayRef<std::pair<llvm::StringRef, mlir::Attribute>> parameters,
             ArrayRef<std::pair<llvm::StringRef, mlir::Value>> ports);
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

  circt::Namespace moduleNameSpace;
  RTLIL::IdString getNewName(StringRef name = "") {
    if (name.empty())
      return NEW_ID;

    return getEscapedName(moduleNameSpace.newName(name));
  }

  LogicalResult lowerPorts();
  LogicalResult lowerBody();

  ExportRTLILModule(ExportRTLILDesign &circuitConverter,
                    Yosys::RTLIL::Module *rtlilModule, hw::HWModuleLike module,
                    bool definedAsBlackBox)
      : circuitConverter(circuitConverter), rtlilModule(rtlilModule),
        module(module), definedAsBlackBox(definedAsBlackBox) {}

  ExportRTLILDesign &circuitConverter;

  DenseMap<Value, RTLIL::Wire *> mapping;
  struct MemoryInfo {
    RTLIL::Memory *mem;
    unsigned portId;
  };
  DenseMap<seq::FirMemOp, MemoryInfo> memoryMapping;
  SmallVector<RTLIL::Wire *> outputs;
  const bool definedAsBlackBox;

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
  hw::HWModuleLike module;
  using hw::TypeOpVisitor<ExportRTLILModule, LogicalResult>::visitTypeOp;
  using hw::StmtVisitor<ExportRTLILModule, LogicalResult>::visitStmt;
  using comb::CombinationalVisitor<ExportRTLILModule, LogicalResult>::visitComb;
  using seq::SeqOpVisitor<ExportRTLILModule, LogicalResult>::visitSeq;

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

  FailureOr<RTLIL::Const> getParameter(Attribute attr);

  // HW type op.
  RTLIL::Const getConstant(IntegerAttr attr);
  RTLIL::Const getConstant(const APInt &attr);

  LogicalResult visitTypeOp(ConstantOp op) {
    return setLowering(op, getConstant(op.getValueAttr()));
  }

  // HW stmt op.
  LogicalResult visitStmt(OutputOp op);
  LogicalResult visitStmt(InstanceOp op);

  // HW expr ops.
  LogicalResult visitTypeOp(AggregateConstantOp op);
  LogicalResult visitTypeOp(ArrayCreateOp op);
  LogicalResult visitTypeOp(ArrayGetOp op);
  LogicalResult visitTypeOp(ArrayConcatOp op);

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
  LogicalResult visitComb(ParityOp op);

  // Seq op.
  LogicalResult visitSeq(seq::FirRegOp op);
  LogicalResult visitSeq(seq::FirMemOp op);
  LogicalResult visitSeq(seq::FirMemWriteOp op);
  LogicalResult visitSeq(seq::FirMemReadOp op);
  LogicalResult visitSeq(seq::FirMemReadWriteOp op);
  LogicalResult visitSeq(seq::FromClockOp op);
  LogicalResult visitSeq(seq::ToClockOp op);

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

  template <typename UnaryFn>
  LogicalResult emitUnaryOp(Operation *op, UnaryFn fn) {
    assert(op->getNumOperands() == 1 && "only expect unary op");
    auto input = getValue(op->getOperand(0));
    if (failed(input))
      return failure();
    return setLowering(op->getResult(0), fn(getNewName(), input.value()));
  }
};

struct ExportRTLILDesign {
  ExportRTLILDesign(Yosys::RTLIL::Design *design,
                    hw::InstanceGraph *instanceGraph)
      : design(design), instanceGraph(instanceGraph) {}
  llvm::DenseMap<StringAttr, Yosys::RTLIL::Module *> moduleMapping;
  LogicalResult addModule(hw::HWModuleLike op, bool defineAsBlackBox = false);
  LogicalResult run();

  SmallVector<std::unique_ptr<ExportRTLILModule>> converter;
  Yosys::RTLIL::Design *design;
  hw::InstanceGraph *instanceGraph;
};
} // namespace

RTLIL::Const ExportRTLILModule::getConstant(const APInt &value) {
  auto width = value.getBitWidth();
  if (width <= 32)
    return RTLIL::Const(value.getZExtValue(), value.getBitWidth());

  // TODO: Use more efficient encoding.
  std::vector<bool> result(width, false);
  for (size_t i = 0; i < width; ++i)
    result[i] = value[i];

  return RTLIL::Const(result);
}

RTLIL::Const ExportRTLILModule::getConstant(IntegerAttr attr) {
  return getConstant(attr.getValue());
}

FailureOr<RTLIL::Const> ExportRTLILModule::getParameter(Attribute attr) {
  return TypeSwitch<Attribute, FailureOr<RTLIL::Const>>(attr)
      .Case<IntegerAttr>([&](auto a) { return getConstant(a); })
      .Case<StringAttr>(
          [&](StringAttr a) { return RTLIL::Const(a.getValue().str()); })
      .Default([](auto) { return failure(); });
}

LogicalResult ExportRTLILModule::lowerPorts() {
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
      if (!definedAsBlackBox)
        setValue(module.getBodyBlock()->getArgument(inputPos++), wire);
      wire->port_input = true;
    } else {
      return module.emitError() << "inout is unssuported";
    }
  }
  // Need to call fixup ports after port mutations.
  rtlilModule->fixup_ports();
  return success();
}

LogicalResult ExportRTLILModule::lowerBody() {
  if (module
          .walk([this](Operation *op) {
            for (auto result : op->getResults()) {
              // TODO: Use SSA name.
              StringAttr name = {};
              if (getBitWidthSeq(result.getType()) >= 0)
                if (failed(createAndSetWire(result, name)))
                  return WalkResult::interrupt();
              return WalkResult::advance();
            }
          })
          .wasInterrupted())
    return failure();

  auto result =
      module
          .walk([this](Operation *op) {
            if (module == op)
              return WalkResult::advance();

            if (isa<comb::CombDialect, hw::HWDialect, seq::SeqDialect>(
                    op->getDialect())) {
              LLVM_DEBUG(llvm::dbgs() << "Visiting " << *op << "\n");
              if (failed(visitOp(op))) {
                op->emitError() << "lowering failed";
                return WalkResult::interrupt();
              }
              LLVM_DEBUG(llvm::dbgs() << "Success \n");
            } else {
              // Ignore Verif, LTL and Sim etc.
            }
            return WalkResult::advance();
          })
          .wasInterrupted();
  return LogicalResult::success(!result);
}

//===----------------------------------------------------------------------===//
// HW Ops.
//===----------------------------------------------------------------------===//

LogicalResult ExportRTLILModule::visitStmt(OutputOp op) {
  assert(op.getNumOperands() == outputs.size());
  for (auto [wire, op] : llvm::zip(outputs, op.getOperands())) {
    auto result = getValue(op);
    if (failed(result))
      return failure();
    rtlilModule->connect(Yosys::SigSpec(wire), result.value());
  }

  return success();
}

LogicalResult ExportRTLILModule::visitStmt(InstanceOp op) {
  // Ignore bound.
  if (op->hasAttr("doNotPrint") || op.getNumResults() == 0)
    return success();
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

  auto *cell = rtlilModule->addCell(getNewName(op.getInstanceName()), id);
  auto connect = [&](ArrayAttr names, auto values) -> LogicalResult {
    for (auto [portName, value] :
         llvm::zip(names.getAsRange<StringAttr>(), values)) {
      auto loweredValue = getValue(value);
      if (failed(loweredValue)) {
        return op.emitError() << "port " << portName << " wasnot lowered";
      }
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

LogicalResult ExportRTLILModule::visitTypeOp(AggregateConstantOp op) {
  SigSpec ret;
  SmallVector<Attribute> worklist{op.getFieldsAttr()};
  while (!worklist.empty()) {
    auto val = worklist.pop_back_val();
    if (auto array = dyn_cast<ArrayAttr>(val))
      for (auto a : llvm::reverse(array))
        worklist.push_back(a);
    else if (auto intVal = dyn_cast<IntegerAttr>(val)) {
      ret.append(getConstant(intVal));
    }
  }
  return success();
}

LogicalResult ExportRTLILModule::visitTypeOp(ArrayCreateOp op) {
  SigSpec ret;
  for (auto operand : llvm::reverse(op.getOperands())) {
    auto result = getValue(operand);
    if (failed(result))
      return result;
    ret.append(result.value());
  }
  return setLowering(op, ret);
}

LogicalResult ExportRTLILModule::visitTypeOp(ArrayGetOp op) {
  auto input = getValue(op.getInput());
  auto index = getValue(op.getIndex());
  auto result = getValue(op);
  if (failed(input) || failed(index) || failed(result))
    return failure();

  auto width =
      hw::type_cast<hw::ArrayType>(op.getInput().getType()).getNumElements();
  auto sig = rtlilModule->Mul(
      NEW_ID, index.value(),
      getConstant(
          APInt(1ll << op.getIndex().getType().getIntOrFloatBitWidth(), width)),
      false);

  (void)rtlilModule->addShiftx(NEW_ID, input.value(), sig, result.value());
  return success();
}

LogicalResult ExportRTLILModule::visitTypeOp(ArrayConcatOp op) {
  SigSpec ret;
  // The order is opposite so reverse it.
  for (auto operand : llvm::reverse(op.getOperands())) {
    auto result = getValue(operand);
    if (failed(result))
      return result;
    ret.append(result.value());
  }
  return setLowering(op, ret);
}

//===----------------------------------------------------------------------===//
// Comb Ops.
//===----------------------------------------------------------------------===//

LogicalResult ExportRTLILModule::visitComb(AddOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Add(name, l, r);
  });
}

LogicalResult ExportRTLILModule::visitComb(SubOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Sub(name, l, r);
  });
}

LogicalResult ExportRTLILModule::visitComb(MuxOp op) {
  auto cond = getValue(op.getCond());
  auto high = getValue(op.getTrueValue());
  auto low = getValue(op.getFalseValue());
  if (failed(cond) || failed(high) || failed(low))
    return failure();

  return setLowering(op, rtlilModule->Mux(getNewName(), low.value(),
                                          high.value(), cond.value()));
}

LogicalResult ExportRTLILModule::visitComb(AndOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->And(name, l, r);
  });
}

LogicalResult ExportRTLILModule::visitComb(MulOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Mul(name, l, r);
  });
}

LogicalResult ExportRTLILModule::visitComb(OrOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Or(name, l, r);
  });
}

LogicalResult ExportRTLILModule::visitComb(XorOp op) {
  return emitVariadicOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Xor(name, l, r);
  });
}

LogicalResult ExportRTLILModule::visitComb(ExtractOp op) {
  auto result = getValue(op.getOperand());
  if (failed(result))
    return result;
  auto sig = result.value().extract(op.getLowBit(),
                                    op.getType().getIntOrFloatBitWidth());
  return setLowering(op, sig);
}

LogicalResult ExportRTLILModule::visitComb(ConcatOp op) {
  SigSpec ret;
  for (auto operand : llvm::reverse(op.getOperands())) {
    auto result = getValue(operand);
    if (failed(result))
      return result;
    ret.append(result.value());
  }
  // TODO: Check endian.
  return setLowering(op, ret);
}

LogicalResult ExportRTLILModule::visitComb(ShlOp op) {
  return emitBinaryOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Shl(name, l, r);
  });
}

LogicalResult ExportRTLILModule::visitComb(ShrUOp op) {
  return emitBinaryOp(op, [&](auto name, auto l, auto r) {
    return rtlilModule->Shr(name, l, r);
  });
}
LogicalResult ExportRTLILModule::visitComb(ShrSOp op) {
  return emitBinaryOp(op, [&](auto name, auto l, auto r) {
    // TODO: Make sure it's correct
    return rtlilModule->Sshr(name, l, r);
  });
}

LogicalResult ExportRTLILModule::visitComb(ReplicateOp op) {
  auto value = getValue(op.getOperand());
  if (failed(value))
    return failure();
  return setLowering(op, value.value().repeat(op.getMultiple()));
}

LogicalResult ExportRTLILModule::visitComb(ParityOp op) {
  return emitUnaryOp(op, [&](auto name, auto input) {
    return rtlilModule->ReduceXor(name, input);
  });
}

LogicalResult ExportRTLILModule::visitComb(ICmpOp op) {
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

LogicalResult ExportRTLILModule::visitSeq(seq::FirRegOp op) {
  auto result = getValue(op.getResult());
  auto clock = getValue(op.getClk());
  auto next = getValue(op.getNext());

  if (failed(result) || failed(clock) || failed(next))
    return failure();

  if (op.getReset()) {
    // addSdff
    auto reset = getValue(op.getReset());
    if (failed(reset))
      return reset;

    auto constOp = op.getResetValue().getDefiningOp<hw::ConstantOp>();
    if (op.getIsAsync()) {
      // Adlatch.
      if (!constOp)
        return failure();
      rtlilModule->addSdff(getNewName(op.getName()), clock.value(),
                           reset.value(), next.value(), result.value(),
                           getConstant(constOp.getValueAttr()));
      return success();
    }
    if (constOp) {
      rtlilModule->addSdff(getNewName(op.getName()), clock.value(),
                           reset.value(), next.value(), result.value(),
                           getConstant(constOp.getValueAttr()));
      return success();
    }
    return op.emitError() << "lowering for non-constant reset value is "
                             "currently not implemented";
  }

  rtlilModule->addDff(getNewName(op.getName()), clock.value(), next.value(),
                      result.value());
  return success();
}

LogicalResult ExportRTLILModule::visitSeq(seq::FirMemOp op) {
  auto *mem = new RTLIL::Memory();
  mem->width = op.getType().getWidth();
  mem->size = op.getType().getDepth();
  mem->name = getNewName("Memory");
  rtlilModule->memories[mem->name] = mem;
  MemoryInfo memInfo;
  memInfo.mem = mem;
  memInfo.portId = 0;
  memoryMapping.insert({op, memInfo});
  return success();
}

FailureOr<RTLIL::Cell *> ExportRTLILModule::createCell(
    llvm::StringRef cellName, llvm::StringRef instanceName,
    ArrayRef<std::pair<StringRef, mlir::Attribute>> parameters,
    ArrayRef<std::pair<StringRef, mlir::Value>> ports) {
  auto *cell =
      rtlilModule->addCell(getNewName(instanceName), getEscapedName(cellName));
  for (auto [portName, value] : ports) {
    auto loweredValue = getValue(value);
    if (failed(loweredValue))
      return failure();
    cell->connections_.insert({getEscapedName(portName), loweredValue.value()});
  }
  for (auto [portName, value] : parameters) {
    auto loweredValue = getParameter(value);
    if (failed(loweredValue))
      return failure();
    cell->parameters.insert({getEscapedName(portName), loweredValue.value()});
  }
  return cell;
}

LogicalResult ExportRTLILModule::visitSeq(seq::FirMemWriteOp op) {
  // cell $memwr_v2 $auto$proc_memwr.cc:45:proc_memwr$49
  //   parameter \ABITS 5
  //   parameter \CLK_ENABLE 1'1
  //   parameter \CLK_POLARITY 1'1
  //   parameter \MEMID "\\Memory"
  //   parameter \PORTID 1
  //   parameter \PRIORITY_MASK 1'1
  //   parameter \WIDTH 65
  //   connect \ADDR $1$memwr$\Memory$mem.sv:29$2_ADDR[4:0]$15
  //   connect \CLK \W0_clk
  //   connect \DATA $1$memwr$\Memory$mem.sv:29$2_DATA[64:0]$16
  //   connect \EN $1$memwr$\Memory$mem.sv:29$2_EN[64:0]$17
  // end
  auto firmem = op.getMemory().getDefiningOp<seq::FirMemOp>();
  if (!firmem)
    return failure();
  SmallVector<std::pair<llvm::StringRef, mlir::Value>> ports{
      {"ADDR", op.getAddress()},
      {"CLK", op.getClk()},
      {"DATA", op.getData()},
      {"EN", op.getData()}};
  OpBuilder builder(module.getContext());
  auto trueConst = builder.getIntegerAttr(builder.getI1Type(), 1);
  auto widthConst = builder.getI32IntegerAttr(
      llvm::Log2_64_Ceil(firmem.getType().getDepth()));

  auto it = memoryMapping.find(firmem);
  assert(it != memoryMapping.end() && "firmem should be visited");
  auto memName = builder.getStringAttr(it->second.mem->name.str());
  auto portId = builder.getI32IntegerAttr(++it->second.portId);
  auto width = builder.getI32IntegerAttr(firmem.getType().getWidth());
  SmallVector<std::pair<llvm::StringRef, Attribute>> parameters{
      {"ABITS", widthConst},       {"CLK_ENABLE", trueConst},
      {"CLK_POLARITY", trueConst}, {"MEMID", memName},
      {"PORTID", portId},          {"WIDTH", width},
      {"PRIORITY_MASK", trueConst}};

  auto cell = createCell("$memwr_v2", "mem_write", parameters, ports);
  if (failed(cell))
    return cell;

  if (op.getEnable()) {
    auto enable = getValue(op.getEnable());
  }

  return success();
}
LogicalResult ExportRTLILModule::visitSeq(seq::FirMemReadOp op) {
  // rtlilModule->addCell("$memrd", id)
  // Memrd
  // cell $memrd
  //   parameter \ABITS 5
  //   parameter \CLK_ENABLE 0
  //   parameter \CLK_POLARITY 0
  //   parameter \MEMID "\\Memory"
  //   parameter \TRANSPARENT 0
  //   parameter \WIDTH 65
  //   connect \ADDR \R0_addr
  //   connect \CLK 1'x
  //   connect \DATA $memrd$\Memory$mem.sv:45$18_DATA
  //   connect \EN 1'x
  // end
  auto firmem = op.getMemory().getDefiningOp<seq::FirMemOp>();
  if (!firmem)
    return failure();
  SmallVector<std::pair<llvm::StringRef, mlir::Value>> ports{
      {"ADDR", op.getAddress()}, {"CLK", op.getClk()}, {"DATA", op.getData()}};
  if (op.getEnable())
    ports.emplace_back("EN", op.getEnable());
  OpBuilder builder(module.getContext());
  auto trueConst = builder.getIntegerAttr(builder.getI1Type(), 1);
  auto falseConst = builder.getIntegerAttr(builder.getI1Type(), 0);

  auto widthConst = builder.getI32IntegerAttr(
      llvm::Log2_64_Ceil(firmem.getType().getDepth()));

  auto it = memoryMapping.find(firmem);
  assert(it != memoryMapping.end() && "firmem should be visited");
  auto memName = builder.getStringAttr(it->second.mem->name.str());
  auto width = builder.getI32IntegerAttr(firmem.getType().getWidth());
  SmallVector<std::pair<llvm::StringRef, Attribute>> parameters{
      {"ABITS", widthConst},
      {"CLK_ENABLE", op.getEnable() ? trueConst : falseConst},
      {"CLK_POLARITY", falseConst},
      {"TRANSPARENT", falseConst},
      {"MEMID", memName},
      {"WIDTH", width}};

  auto cell = createCell("$memrd", "mem_read", parameters, ports);
  if (failed(cell))
    return cell;
  if (!op.getEnable()) {
    (*cell)->connections_.insert(
        {getEscapedName("EN"), SigSpec(getConstant(APInt(1, 1)))});
  }

  return success();
}
LogicalResult ExportRTLILModule::visitSeq(seq::FirMemReadWriteOp op) {
  return failure();
}

LogicalResult ExportRTLILModule::visitSeq(seq::FromClockOp op) {
  auto result = getValue(op.getInput());
  if (failed(result))
    return result;
  return setLowering(op, result.value());
}

LogicalResult ExportRTLILModule::visitSeq(seq::ToClockOp op) {
  auto result = getValue(op.getInput());
  if (failed(result))
    return result;
  return setLowering(op, result.value());
}

void circt::rtlil::init_yosys(bool enableLog) {
  // Set up yosys.
  Yosys::log_streams.clear();
  if (enableLog)
    Yosys::log_streams.push_back(&std::cerr);
  Yosys::log_error_stderr = true;
  Yosys::yosys_setup();
}

LogicalResult ExportRTLILDesign::addModule(hw::HWModuleLike op,
                                           bool defineAsBlackBox) {
  defineAsBlackBox |= isa<hw::HWModuleExternOp>(op);
  if (design->has(getEscapedName(op.getModuleName()))) {
    return success(defineAsBlackBox);
  }
  auto *newModule = design->addModule(getEscapedName(op.getModuleName()));
  if (!moduleMapping.insert({op.getModuleNameAttr(), newModule}).second)
    return failure();
  if (defineAsBlackBox)
    newModule->set_bool_attribute(ID::blackbox);
  converter.emplace_back(std::make_unique<ExportRTLILModule>(
      *this, newModule, op, defineAsBlackBox));
  return converter.back()->lowerPorts();
}

LogicalResult ExportRTLILDesign::run() {
  for (auto &c : converter)
    if (!c->definedAsBlackBox && failed(c->lowerBody()))
      return failure();
  return success();
}

mlir::FailureOr<std::unique_ptr<Yosys::RTLIL::Design>>
circt::rtlil::exportRTLILDesign(ArrayRef<hw::HWModuleLike> modules,
                                ArrayRef<hw::HWModuleLike> blackBox,
                                hw::InstanceGraph &instanceGraph) {
  auto theDesign = std::make_unique<Yosys::RTLIL::Design>();
  auto *design = theDesign.get();
  ExportRTLILDesign exporter(design, &instanceGraph);

  for (auto op : modules)
    if (failed(exporter.addModule(op)))
      return failure();

  for (auto op : blackBox) {
    if (failed(exporter.addModule(op, true)))
      return failure();
  }

  if (failed(exporter.run()))
    return failure();

  return std::move(theDesign);
}
