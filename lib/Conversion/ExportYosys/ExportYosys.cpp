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

#include "kernel/rtlil.h"
#include "kernel/yosys.h"

#define DEBUG_TYPE "export-yosys"

using namespace circt;
using namespace hw;
using namespace comb;
using namespace Yosys;

namespace {
#define GEN_PASS_DEF_EXPORTYOSYS
#define GEN_PASS_DEF_EXPORTYOSYSPARALLEL
#include "circt/Conversion/Passes.h.inc"

struct ExportYosysPass : public impl::ExportYosysBase<ExportYosysPass> {
  void runOnOperation() override;
};
struct ExportYosysParallelPass
    : public impl::ExportYosysParallelBase<ExportYosysParallelPass> {
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

  ModuleConverter(YosysCircuitImporter &circuitConverter,
                  Yosys::RTLIL::Module *rtlilModule, hw::HWModuleLike module,
                  bool definedAsBlackBox)
      : circuitConverter(circuitConverter), rtlilModule(rtlilModule),
        module(module), definedAsBlackBox(definedAsBlackBox) {}

  YosysCircuitImporter &circuitConverter;

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

struct YosysCircuitImporter {
  YosysCircuitImporter(Yosys::RTLIL::Design *design,
                       InstanceGraph *instanceGraph)
      : design(design), instanceGraph(instanceGraph) {}
  llvm::DenseMap<StringAttr, Yosys::RTLIL::Module *> moduleMapping;

  LogicalResult addModule(hw::HWModuleLike op, bool defineAsBlackBox = false) {
    defineAsBlackBox |= isa<hw::HWModuleExternOp>(op);
    if (design->has(getEscapedName(op.getModuleName()))) {
      return success(defineAsBlackBox);
    }
    auto *newModule = design->addModule(getEscapedName(op.getModuleName()));
    if (!moduleMapping.insert({op.getModuleNameAttr(), newModule}).second)
      return failure();
    if (defineAsBlackBox)
      newModule->set_bool_attribute(ID::blackbox);
    converter.emplace_back(*this, newModule, op, defineAsBlackBox);
    return converter.back().lowerPorts();
  }

  SmallVector<ModuleConverter> converter;
  LogicalResult run() {
    for (auto &c : converter)
      if (!c.definedAsBlackBox && failed(c.lowerBody()))
        return failure();
    return success();
  }

  Yosys::RTLIL::Design *design;
  InstanceGraph *instanceGraph;
};

struct RTLILImporter {
  RTLIL::Design *design;
  const bool mutateInplace;
  LogicalResult run(mlir::ModuleOp module);
  using ModuleMappingTy = DenseMap<StringAttr, hw::HWModuleLike>;
  ModuleMappingTy moduleMapping;
};

struct RTLILModuleImporter {
  RTLIL::Module *rtlilModule;
  hw::HWModuleLike module;
  MLIRContext *context;
  hw::HWModuleLike getModuleOp() { return module; }
  RTLILModuleImporter(MLIRContext *context, const RTLILImporter &importer,
                      RTLIL::Module *rtlilModule, OpBuilder &moduleBuilder)
      : importer(importer), rtlilModule(rtlilModule), context(context) {
    builder = std::make_unique<OpBuilder>(context);
    block = std::make_unique<Block>();
    registerPatterns();
    size_t size = rtlilModule->ports.size();
    SmallVector<hw::PortInfo> ports(size);
    SmallVector<Value> values(size);
    auto modName = getStr(rtlilModule->name);

    size_t numInput = 0, numOutput = 0;
    for (auto port : rtlilModule->ports) {
      auto *wire = rtlilModule->wires_[port];
      assert(wire->port_input || wire->port_output);

      size_t portId = wire->port_id - 1;
      size_t argNum = (wire->port_output ? numOutput : numInput)++;
      ports[portId].name = getStr(wire->name);
      ports[portId].argNum = argNum;
      ports[portId].type = builder->getIntegerType(wire->width);
      ports[portId].dir =
          wire->port_output ? hw::ModulePort::Output : hw::ModulePort::Input;
    }
    if (rtlilModule->get_blackbox_attribute()) {
      module = moduleBuilder.create<hw::HWModuleExternOp>(
          UnknownLoc::get(context), modName, ports);
      module.setPrivate();
    } else
      module = moduleBuilder.create<hw::HWModuleOp>(UnknownLoc::get(context),
                                                    modName, ports);
  }
  const RTLILImporter &importer;
  StringAttr getStr(const Yosys::RTLIL::IdString &str) const {
    StringRef s(str.c_str());
    return builder->getStringAttr(s.starts_with("\\") ? s.drop_front(1) : s);
  }

  mlir::TypedValue<hw::InOutType> getInOutValue(Location loc,
                                                SigSpec &sigSpec) {
    if (sigSpec.is_wire())
      return wireMapping.lookup(getStr(sigSpec.as_wire()->name));

    // Const cannot be inout.
    if (sigSpec.is_fully_const()) {
      return {};
    }
    // Bit selection.
    if (sigSpec.is_bit()) {
      auto bit = sigSpec.as_bit();
      if (!bit.wire) {
        module.emitError() << "is not wire";
        return {};
      }
      auto v = wireMapping.lookup(getStr(bit.wire->name));
      assert(v);
      auto width = 1;
      auto offset = bit.offset;
      auto idx = builder->create<hw::ConstantOp>(
          v.getLoc(),
          APInt(bit.wire->width == 0 ? 1 : llvm::Log2_32_Ceil(bit.wire->width),
                bit.offset));
      return builder->create<sv::ArrayIndexInOutOp>(v.getLoc(), v, idx);
    }

    // Range selection.
    if (sigSpec.is_chunk()) {
      auto chunk = sigSpec.as_chunk();
      if (!chunk.wire) {
        mlir::emitError(loc) << "unsupported chunk states";
        return {};
      }

      auto v = wireMapping.lookup(getStr(chunk.wire->name));
      auto arrayLength =
          cast<hw::ArrayType>(v.getElementType()).getNumElements();
      auto idx = builder->create<hw::ConstantOp>(
          v.getLoc(),
          APInt(arrayLength == 0 ? 1 : llvm::Log2_32_Ceil(arrayLength),
                chunk.offset));
      return builder->create<sv::IndexedPartSelectInOutOp>(loc, v, idx,
                                                           chunk.width);
    }

    mlir::emitError(loc) << "unsupported lhs value";
    return {};
  }
  LogicalResult connect(Location loc, mlir::TypedValue<hw::InOutType> lhs,
                        Value rhs) {
    if (lhs.getType().getElementType() != rhs.getType())
      rhs = builder->create<hw::BitcastOp>(loc, lhs.getType().getElementType(),
                                           rhs);
    builder->create<sv::AssignOp>(loc, lhs, rhs);
    return success();
  }

  LogicalResult connect(Location loc, SigSpec &lhs, SigSpec &rhs) {
    auto lhsValue = getInOutValue(loc, lhs);
    Value output = convertSigSpec(rhs);
    if (!lhsValue || !output) {
      return mlir::emitError(loc) << "unsupported connection";
    }

    return connect(loc, lhsValue, output);
  }

  DenseMap<StringAttr, sv::WireOp> wireMapping;

  // mlir::TypedValue<hw::InOutType> getInout();
  LogicalResult
  importBody(const RTLILImporter::ModuleMappingTy &moduleMapping) {
    if (isa<HWModuleExternOp>(module))
      return success();

    SmallVector<std::pair<size_t, Value>> outputs;
    builder->setInsertionPointToStart(module.getBodyBlock());
    // Init wires.
    ModulePortInfo portInfo(module.getPortList());
    for (auto wire : rtlilModule->wires()) {
      if (wire->port_input) {
        auto arg = module.getBodyBlock()->getArgument(
            portInfo.at(wire->port_id - 1).argNum);
        mapping.insert({getStr(wire->name), arg});
      } else {
        auto loc = builder->getUnknownLoc();
        auto w = builder->create<sv::WireOp>(
            loc, hw::ArrayType::get(builder->getIntegerType(1), wire->width));
        auto read = builder->create<hw::BitcastOp>(
            loc, builder->getIntegerType(wire->width),
            builder->create<sv::ReadInOutOp>(loc, w));
        mapping.insert({getStr(wire->name), read});
        wireMapping.insert({getStr(wire->name), w});
        if (wire->port_output) {
          outputs.emplace_back(wire->port_id - 1, read);
        }
      }
    }

    llvm::sort(
        outputs.begin(), outputs.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    SmallVector<Value> results;
    for (auto p : llvm::map_range(outputs, [](auto &p) { return p.second; }))
      results.push_back(p);

    // Ensure terminator.
    module.getBodyBlock()->getTerminator()->setOperands(results);
    builder->setInsertionPointToStart(module.getBodyBlock());

    // Import connections.
    for (auto con : rtlilModule->connections()) {
      if (failed(connect(builder->getUnknownLoc(), con.first, con.second)))
        return failure();
    }

    // Import cells.
    for (auto cell : rtlilModule->cells()) {
      if (failed(importCell(moduleMapping, cell)))
        return failure();
    }

    return success();
  }

  Value getValueForWire(const RTLIL::Wire *wire) const {
    return mapping.at(getStr(wire->name));
  }

  Value convertSigSpec(const RTLIL::SigSpec &sigSpec) {
    if (sigSpec.is_wire())
      return getValueForWire(sigSpec.as_wire());
    if (sigSpec.is_fully_undef()) {
      return builder->create<sv::ConstantXOp>(
          builder->getUnknownLoc(),
          builder->getIntegerType(sigSpec.as_const().size()));
    }
    if (sigSpec.is_fully_const()) {
      if (sigSpec.as_const().size() > 32)
        return Value();
      return builder->create<hw::ConstantOp>(
          builder->getUnknownLoc(),
          APInt(sigSpec.as_const().size(), sigSpec.as_const().as_int(false)));
    }
    if (sigSpec.is_bit()) {
      auto bit = sigSpec.as_bit();
      if (!bit.wire) {
        module.emitError() << "is not wire";
        return {};
      }
      auto v = getValueForWire(bit.wire);
      auto width = 1;
      auto offset = bit.offset;
      return builder->create<comb::ExtractOp>(v.getLoc(), v, offset, width);
    }
    if (sigSpec.is_chunk()) {
      auto bit = sigSpec.as_bit();
      if (!bit.wire) {
        module.emitError() << "is not wire";
        return {};
      }
      auto v = getValueForWire(bit.wire);
      auto width = sigSpec.as_chunk().width;
      auto offset = sigSpec.as_chunk().offset;
      return builder->create<comb::ExtractOp>(v.getLoc(), v, offset, width);
    }

    SmallVector<mlir::Value> chunks;
    for (auto w : sigSpec.chunks())
      chunks.push_back(convertSigSpec(w));
    return builder->create<comb::ConcatOp>(builder->getUnknownLoc(), chunks);
  }

  Value getPortValue(RTLIL::Cell *cell, StringRef portName) {
    return convertSigSpec(cell->getPort(getEscapedName(portName)));
  }
  SigSpec getPortSig(RTLIL::Cell *cell, StringRef portName) {
    return cell->getPort(getEscapedName(portName));
  }

  class CellPatternBase {
  public:
    CellPatternBase(StringRef typeName, ArrayRef<StringRef> inputPortNames,
                    StringRef outputPortName)
        : typeName(typeName), inputPortNames(inputPortNames),
          outputPortName(outputPortName){};
    LogicalResult convert(RTLILModuleImporter &importer, Cell *cell) {
      SmallVector<Value> inputs;
      for (auto name : inputPortNames) {
        inputs.push_back(importer.getPortValue(cell, name));
        if (!inputs.back())
          return failure();
      }
      auto location = importer.builder->getUnknownLoc();

      auto rhsValue = convert(*importer.builder, location, inputs);
      auto lhsSig = importer.getPortSig(cell, outputPortName);
      auto lhsValue = importer.getInOutValue(location, lhsSig);
      return importer.connect(location, lhsValue, rhsValue);
    }

  private:
    virtual Value convert(OpBuilder &builder, Location location,
                          ValueRange inputValues) = 0;
    SmallString<4> typeName;
    SmallVector<SmallString<4>> inputPortNames;
    SmallString<4> outputPortName;
  };
  template <typename OpName> struct CellOpPattern : public CellPatternBase {
  public:
    using CellPatternBase::CellPatternBase;
    Value convert(OpBuilder &builder, Location location,
                  ValueRange inputValues) override {
      return builder.create<OpName>(location, inputValues, false);
    }
  };

  template <bool isAnd> struct AndOrNotOpPattern : public CellPatternBase {
  public:
    using CellPatternBase::CellPatternBase;
    Value convert(OpBuilder &builder, Location location,
                  ValueRange inputValues) override {
      auto notB =
          comb::createOrFoldNot(location, inputValues[1], builder, false);

      Value value;
      if (isAnd)
        value = builder.create<AndOp>(
            location, ArrayRef<Value>{inputValues[0], notB}, false);
      else
        value = builder.create<OrOp>(
            location, ArrayRef<Value>{inputValues[0], notB}, false);
      return value;
    }
  };
  struct NorOpPattern : public CellPatternBase {
  public:
    using CellPatternBase::CellPatternBase;
    Value convert(OpBuilder &builder, Location location,
                  ValueRange inputValues) override {
      auto aOrB = builder.create<OrOp>(location, inputValues, false);
      return comb::createOrFoldNot(location, aOrB, builder, false);
    }
  };
  struct NandOpPattern : public CellPatternBase {
  public:
    using CellPatternBase::CellPatternBase;
    Value convert(OpBuilder &builder, Location location,
                  ValueRange inputValues) override {
      auto aOrB = builder.create<AndOp>(location, inputValues, false);
      return comb::createOrFoldNot(location, aOrB, builder, false);
    }
  };
  struct NotOpPattern : public CellPatternBase {
  public:
    using CellPatternBase::CellPatternBase;
    Value convert(OpBuilder &builder, Location location,
                  ValueRange inputValues) override {
      return comb::createOrFoldNot(location, inputValues[0], builder, false);
    }
  };
  struct MuxOpPattern : public CellPatternBase {
  public:
    using CellPatternBase::CellPatternBase;
    Value convert(OpBuilder &builder, Location location,
                  ValueRange inputValues) override {
      return builder.create<MuxOp>(location, inputValues[2], inputValues[1],
                                   inputValues[0]);
    }
  };

  struct XnorOpPattern : public CellPatternBase {
  public:
    using CellPatternBase::CellPatternBase;
    Value convert(OpBuilder &builder, Location location,
                  ValueRange inputValues) override {
      auto aAndB = builder.create<AndOp>(location, inputValues, false);
      auto notA =
          comb::createOrFoldNot(location, inputValues[0], builder, false);
      auto notB =
          comb::createOrFoldNot(location, inputValues[1], builder, false);

      auto notAnds =
          builder.create<AndOp>(location, ArrayRef<Value>{notA, notB}, false);

      return builder.create<OrOp>(location, ArrayRef<Value>{aAndB, notAnds},
                                  false);
    }
  };

  llvm::StringMap<std::unique_ptr<CellPatternBase>> handler;
  template <typename CellPattern>
  void addPattern(StringRef typeName, ArrayRef<StringRef> inputPortNames,
                  StringRef outputPortName) {
    handler.insert({typeName, std::make_unique<CellPattern>(
                                  typeName, inputPortNames, outputPortName)});
  }

  template <typename OpName>
  void addOpPattern(StringRef typeName, ArrayRef<StringRef> inputPortNames,
                    StringRef outputPortName) {
    handler.insert({typeName, std::make_unique<CellOpPattern<OpName>>(
                                  typeName, inputPortNames, outputPortName)});
  }

  void registerPatterns() {
    addOpPattern<comb::XorOp>("$_XOR_", {"A", "B"}, "Y");
    addOpPattern<comb::AndOp>("$_AND_", {"A", "B"}, "Y");
    addOpPattern<comb::OrOp>("$_OR_", {"A", "B"}, "Y");
    addPattern<AndOrNotOpPattern</*isAnd=*/true>>("$_ANDNOT_", {"A", "B"}, "Y");
    addPattern<AndOrNotOpPattern</*isAnd=*/false>>("$_ORNOT_", {"A", "B"}, "Y");
    addPattern<XnorOpPattern>("$_XNOR_", {"A", "B"}, "Y");
    addPattern<NorOpPattern>("$_NOR_", {"A", "B"}, "Y");
    addPattern<NandOpPattern>("$_NAND_", {"A", "B"}, "Y");
    addPattern<NotOpPattern>("$_NOT_", {"A"}, "Y");
    addPattern<MuxOpPattern>("$_MUX_", {"A", "B", "S"}, "Y");
  }

  LogicalResult importCell(const RTLILImporter::ModuleMappingTy &moduleMapping,
                           RTLIL::Cell *cell) {
    auto v = getStr(cell->type);
    auto mod = rtlilModule->design->module(cell->type);
    LLVM_DEBUG(llvm::dbgs() << "Importing Cell " << v << "\n";);
    // Standard cells.
    auto it = handler.find(v);
    auto location = builder->getUnknownLoc();

    if (cell->parameters.size()) {
      mlir::emitWarning(location)
          << "parameters on a cell is currently dropped";
    }
    if (it == handler.end()) {

      SmallVector<mlir::TypedValue<hw::InOutType>> lhsValues;
      SmallVector<Value> values;
      SmallVector<hw::PortInfo> ports;
      Operation *referredModule;
      for (auto [lhs, rhs] : cell->connections()) {
        hw::PortInfo hwPort;
        hwPort.name = getStr(lhs);
        if (cell->output(lhs)) {
          hwPort.dir = hw::PortInfo::Output;
          lhsValues.push_back(getInOutValue(location, rhs));
          if (!lhsValues.back())
            return mlir::emitError(location)
                   << "port lowering failed cell name=" << v
                   << " port name=" << lhs.c_str();
          hwPort.type = lhsValues.back().getType().getElementType();
        } else {
          hwPort.dir = hw::PortInfo::Input;
          values.push_back(convertSigSpec(rhs));

          if (!values.back())
            return mlir::emitError(location)
                   << "port lowering failed cell name=" << v
                   << " port name=" << lhs.c_str();

          hwPort.type = values.back().getType();
        }

        ports.push_back(hwPort);
      }

      if (v.getValue().starts_with("$")) {
        // Yosys std cells. Just lower it to external module instances.
        auto &extMod = exeternalModules[v];
        if (!extMod) {
          OpBuilder::InsertionGuard guard(*builder);
          builder->setInsertionPointToStart(block.get());
          extMod = builder->create<hw::HWModuleExternOp>(location, v, ports);
          extMod.setPrivate();
        }
        referredModule = extMod;
      } else {
        // Otherwise lower it to an instance.
        auto mod = moduleMapping.lookup(v);
        if (!mod)
          return failure();
        referredModule = mod;
      }

      auto result = builder->create<hw::InstanceOp>(location, referredModule,
                                                    getStr(cell->name), values);
      assert(result.getNumResults() == lhsValues.size());
      for (auto [lhs, rhs] : llvm::zip(lhsValues, result.getResults()))
        if (failed(connect(location, lhs, (Value)rhs)))
          return failure();
      return success();
    }
    auto result = it->second->convert(*this, cell);
    return result;
  }
  DenseMap<StringAttr, Value> mapping;
  std::unique_ptr<OpBuilder> builder;
  llvm::MapVector<StringAttr, hw::HWModuleExternOp> exeternalModules;
  std::unique_ptr<Block> block;
};
} // namespace

RTLIL::Const ModuleConverter::getConstant(const APInt &value) {
  auto width = value.getBitWidth();
  if (width <= 32)
    return RTLIL::Const(value.getZExtValue(), value.getBitWidth());

  // TODO: Use more efficient encoding.
  std::vector<bool> result(width, false);
  for (size_t i = 0; i < width; ++i)
    result[i] = value[i];

  return RTLIL::Const(result);
}

RTLIL::Const ModuleConverter::getConstant(IntegerAttr attr) {
  return getConstant(attr.getValue());
}

FailureOr<RTLIL::Const> ModuleConverter::getParameter(Attribute attr) {
  return TypeSwitch<Attribute, FailureOr<RTLIL::Const>>(attr)
      .Case<IntegerAttr>([&](auto a) { return getConstant(a); })
      .Case<StringAttr>(
          [&](StringAttr a) { return RTLIL::Const(a.getValue().str()); })
      .Default([](auto) { return failure(); });
}

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

LogicalResult ModuleConverter::lowerBody() {
  module.walk([this](Operation *op) {
    for (auto result : op->getResults()) {
      // TODO: Use SSA name.
      mlir::StringAttr name = {};
      if (getBitWidthSeq(result.getType()) >= 0)
        createAndSetWire(result, name);
    }
  });

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

LogicalResult ModuleConverter::visitTypeOp(AggregateConstantOp op) {
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

LogicalResult ModuleConverter::visitTypeOp(ArrayCreateOp op) {
  SigSpec ret;
  for (auto operand : llvm::reverse(op.getOperands())) {
    auto result = getValue(operand);
    if (failed(result))
      return result;
    ret.append(result.value());
  }
  return setLowering(op, ret);
}

LogicalResult ModuleConverter::visitTypeOp(ArrayGetOp op) {
  auto input = getValue(op.getInput());
  auto index = getValue(op.getIndex());
  auto result = getValue(op);
  if (failed(input) || failed(index) || failed(result))
    return failure();

  (void)rtlilModule->addShiftx(NEW_ID, input.value(), index.value(),
                               result.value());
  return success();
}

LogicalResult ModuleConverter::visitTypeOp(ArrayConcatOp op) {
  SigSpec ret;
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

LogicalResult ModuleConverter::visitComb(ParityOp op) {
  return emitUnaryOp(op, [&](auto name, auto input) {
    return rtlilModule->ReduceXor(name, input);
  });
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

LogicalResult ModuleConverter::visitSeq(seq::FirMemOp op) {
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

FailureOr<RTLIL::Cell *> ModuleConverter::createCell(
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

LogicalResult ModuleConverter::visitSeq(seq::FirMemWriteOp op) {
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
LogicalResult ModuleConverter::visitSeq(seq::FirMemReadOp op) {
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
LogicalResult ModuleConverter::visitSeq(seq::FirMemReadWriteOp op) {
  return failure();
}

LogicalResult ModuleConverter::visitSeq(seq::FromClockOp op) {
  auto result = getValue(op.getInput());
  if (failed(result))
    return result;
  return setLowering(op, result.value());
}

LogicalResult ModuleConverter::visitSeq(seq::ToClockOp op) {
  auto result = getValue(op.getInput());
  if (failed(result))
    return result;
  return setLowering(op, result.value());
}

static void init_yosys() {
  // Set up yosys.
  Yosys::log_streams.clear();
  Yosys::log_streams.push_back(&std::cerr);
  Yosys::log_error_stderr = true;
  Yosys::yosys_setup();
}

void ExportYosysPass::runOnOperation() {
  init_yosys();
  auto theDesign = std::make_unique<Yosys::RTLIL::Design>();
  auto *design = theDesign.get();
  auto &theInstanceGraph = getAnalysis<hw::InstanceGraph>();
  YosysCircuitImporter exporter(design, &theInstanceGraph);
  for (auto op : getOperation().getOps<hw::HWModuleLike>()) {
    if (failed(exporter.addModule(op)))
      return signalPassFailure();
  }

  if (failed(exporter.run()))
    return signalPassFailure();

  RTLIL_BACKEND::dump_design(std::cout, design, false);
  // Yosys::run_pass("hierarchy -top DigitalTop", design);
  Yosys::run_pass("synth_xilinx", design);
  Yosys::run_pass("write_rtlil", design);
  Yosys::run_pass("write_verilog synth.v", design);
  // RTLIL_BACKEND::dump_design(std::cout, design, false);
  while (getOperation().begin() != getOperation().end())
    getOperation().begin()->erase();
  RTLILImporter importer{design, true};
  if (failed(importer.run(getOperation())))
    return signalPassFailure();
}

LogicalResult runYosys(Location loc, StringRef inputFilePath,
                       std::string command) {
  auto yosysPath = llvm::sys::findProgramByName("yosys");
  if (!yosysPath) {
    return mlir::emitError(loc) << "cannot find 'yosys' executable. Please add "
                                   "yosys to PATH. Error message='"
                                << yosysPath.getError().message() << "'";
  }
  StringRef commands[] = {"-q", "-p", command, "-f", "rtlil", inputFilePath};
  auto exitCode = llvm::sys::ExecuteAndWait(yosysPath.get(), commands);
  return success(exitCode == 0);
}

static llvm::DenseSet<mlir::StringAttr> designSet(InstanceGraph &instanceGraph,
                                                  StringAttr dut) {
  auto dutModule = instanceGraph.lookup(dut);
  if (!dutModule)
    return {};
  SmallVector<circt::igraph::InstanceGraphNode *, 8> worklist{dutModule};
  DenseSet<StringAttr> visited;
  while (!worklist.empty()) {
    auto *mod = worklist.pop_back_val();
    if (!mod)
      continue;
    if (!visited.insert(mod->getModule().getModuleNameAttr()).second)
      continue;
    for (auto inst : *mod) {
      if (!inst)
        continue;

      igraph::InstanceOpInterface t = inst->getInstance();
      assert(t);
      if (t->hasAttr("doNotPrint") || t->getNumResults() == 0)
        continue;
      worklist.push_back(inst->getTarget());
    }
  }
  return visited;
}

void ExportYosysParallelPass::runOnOperation() {
  // Set up yosys.
  init_yosys();
  auto &theInstanceGraph = getAnalysis<hw::InstanceGraph>();
  auto &table = getAnalysis<mlir::SymbolTable>();

  auto dut = StringAttr::get(&getContext(), "DigitalTop");
  DenseSet<StringAttr> designs;
  if (table.lookup(dut))
    designs = designSet(theInstanceGraph, dut);
  auto isInDesign = [&](StringAttr mod) -> bool {
    if (designs.empty())
      return true;
    return designs.count(mod);
  };
  SmallVector<std::pair<hw::HWModuleOp, std::string>> results;
  for (auto op : getOperation().getOps<hw::HWModuleOp>()) {
    auto theDesign = std::make_unique<Yosys::RTLIL::Design>();
    auto *design = theDesign.get();
    YosysCircuitImporter exporter(design, &theInstanceGraph);
    auto *node = theInstanceGraph.lookup(op.getModuleNameAttr());
    if (!isInDesign(op.getModuleNameAttr()))
      continue;
    if (failed(exporter.addModule(op)))
      return signalPassFailure();

    for (auto instance : *node) {
      auto mod = instance->getTarget()->getModule<hw::HWModuleLike>();
      if (failed(exporter.addModule(mod, true)))
        return signalPassFailure();
    }

    if (failed(exporter.run()))
      return signalPassFailure();

    std::error_code ec;
    SmallString<128> fileName;
    if ((ec = llvm::sys::fs::createTemporaryFile("yosys", "rtlil", fileName))) {
      op.emitError() << ec.message();
      return signalPassFailure();
    }

    results.emplace_back(op, fileName);

    {
      std::ofstream myfile(fileName.c_str());
      RTLIL_BACKEND::dump_design(myfile, design, false);
      myfile.close();
    }
  }
  llvm::sys::SmartMutex<true> mutex;
  auto logger = [&mutex](auto callback) {
    llvm::sys::SmartScopedLock<true> lock(mutex);
    callback();
  };
  if (failed(mlir::failableParallelForEachN(
          &getContext(), 0, results.size(), [&](auto i) {
            auto &[op, test] = results[i];
            logger([&] {
              llvm::errs() << "[yosys-optimizer] Running [" << i + 1 << "/"
                           << results.size() << "] " << op.getModuleName()
                           << "\n";
            });

            auto result =
                runYosys(op.getLoc(), test, "synth_xilinx; write_verilog");

            logger([&] {
              llvm::errs() << "[yosys-optimizer] Finished [" << i + 1 << "/"
                           << results.size() << "] " << op.getModuleName()
                           << "\n";
            });

            // Remove temporary rtlil if success.
            if (succeeded(result))
              llvm::sys::fs::remove(test);
            else
              op.emitError() << "Found error in yosys"
                             << "\n";

            return result;
          })))
    return signalPassFailure();
}

/*
  @ModSym_Synth (in %0.. , output : %1) {
  }

  @FooSym() {
    %2, %3, %4 = @ModSym_Synth()
    @test @other inst
  }
*/

LogicalResult RTLILImporter::run(mlir::ModuleOp module) {
  SmallVector<RTLILModuleImporter> modules;
  OpBuilder builder(module);
  builder.setInsertionPointToStart(module.getBody());
  for (auto mod : design->modules()) {
    modules.emplace_back(module.getContext(), *this, mod, builder);
    auto moduleOp = modules.back().getModuleOp();
    moduleMapping.insert({moduleOp.getNameAttr(), moduleOp});
  }
  llvm::DenseSet<StringAttr> extMap;
  for (auto &mod : modules) {
    if (failed(mod.importBody(moduleMapping)))
      return failure();
    for (auto [str, ext] : mod.exeternalModules) {
      auto it = extMap.insert(str).second;
      if (it) {
        ext->moveBefore(module.getBody(), module.getBody()->begin());
      }
    }
  }

  return success();
}
//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> circt::createExportYosys() {
  return std::make_unique<ExportYosysPass>();
}

std::unique_ptr<mlir::Pass> circt::createExportYosysParallel() {
  return std::make_unique<ExportYosysParallelPass>();
}
