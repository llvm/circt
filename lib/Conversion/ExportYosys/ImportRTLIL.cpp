//===- ImportRTLIL.cpp - RTLIL import implementation ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the import of RTLIL designs into CIRCT.
//
//===----------------------------------------------------------------------===//

#include "RTLILConverterInternal.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"

// Yosys headers.
#include "kernel/rtlil.h"
#include "kernel/yosys.h"

#define DEBUG_TYPE "yosys-optimizer"

using namespace circt;
using namespace hw;
using namespace comb;
using namespace Yosys;
using namespace rtlil;

namespace {
class ImportRTLILModule;

// Base class for cell pattern handling
class CellPatternBase {
public:
  enum ImplicitWidthStrategy {
    HasOperandsAndResultSameWidth,
    OperandMustBeMax,
    None
  };
  CellPatternBase(StringRef typeName, ArrayRef<StringRef> inputPortNames,
                  StringRef outputPortName,
                  ImplicitWidthStrategy implicitWidthStrategy = None)
      : typeName(typeName), inputPortNames(inputPortNames),
        outputPortName(outputPortName),
        implicitWidthStrategy(implicitWidthStrategy){};
  LogicalResult convert(ImportRTLILModule &importer, Cell *cell);
  virtual ~CellPatternBase() {}

private:
  virtual Value convert(Cell *cell, OpBuilder &builder, Location location,
                        ValueRange inputValues) = 0;

  SmallString<4> typeName;
  SmallVector<SmallString<4>> inputPortNames;
  SmallString<4> outputPortName;
  const ImplicitWidthStrategy implicitWidthStrategy;
};

struct ImportRTLILDesign {
  RTLIL::Design *design;
  ImportRTLILDesign(RTLIL::Design *design) : design(design) {}
  LogicalResult run(mlir::ModuleOp module);
  using ModuleMappingTy = DenseMap<StringAttr, hw::HWModuleLike>;
  ModuleMappingTy moduleMapping;
};

static StringRef getStrippedName(const Yosys::RTLIL::IdString &str) {
  StringRef s(str.c_str());
  return s.starts_with("\\") ? s.drop_front(1) : s;
}

// Class for importing a single RTLIL module
class ImportRTLILModule {
public:
  ImportRTLILModule(MLIRContext *context, const ImportRTLILDesign &importer,
                    RTLIL::Module *rtlilModule, OpBuilder &moduleBuilder);

  hw::HWModuleLike getModuleOp() { return module; }
  LogicalResult
  importBody(const ImportRTLILDesign::ModuleMappingTy &moduleMapping);
  const auto &getExternalModules() const { return exeternalModules; }

  friend class CellPatternBase;

  // Get the value of a port of a cell.
  static Value getExtOrTruncPortValue(OpBuilder &builder, Value portVal,

                                      unsigned resultWidth) {
    auto width = hw::getBitWidth(portVal.getType());
    if (width > resultWidth) {
      portVal = builder.create<comb::ExtractOp>(portVal.getLoc(), portVal, 0,
                                                resultWidth);
    } else if (width < resultWidth) {
      portVal = builder.create<comb::ConcatOp>(
          portVal.getLoc(),
          builder.create<hw::ConstantOp>(
              portVal.getLoc(), llvm::APInt::getZero(resultWidth - width)),
          portVal);
    }
    return portVal;
  }

private:
  RTLIL::Module *rtlilModule;
  hw::HWModuleLike module;
  // MLIRContext *context;
  // const ImportRTLILDesign &importer;

  StringAttr getStringAttr(const Yosys::RTLIL::IdString &str) const {
    StringRef s(str.c_str());
    return builder->getStringAttr(getStrippedName(str));
  }

  StringAttr getStringAttr(Yosys::RTLIL::Wire *wire) const {
    return getStringAttr(wire->name);
  }

  mlir::TypedValue<hw::InOutType> getInOutValue(Location loc,
                                                const SigSpec &sigSpec);
  LogicalResult connect(Location loc, mlir::TypedValue<hw::InOutType> lhs,
                        Value rhs);

  LogicalResult connect(Location loc, const SigSpec &lhs, const SigSpec &rhs);

  DenseMap<StringAttr, sv::WireOp> wireMapping;
  sv::WireOp getWireValue(RTLIL::Wire *wire) const {
    return wireMapping.lookup(getStringAttr(wire));
  }

  Value getValueForWire(const RTLIL::Wire *wire) const {
    return mapping.at(getStringAttr(wire->name));
  }

  IntegerAttr getIntegerAttr(const RTLIL::Const &c) {
    APInt a = APInt::getZero(c.size());
    for (auto [idx, b] : llvm::enumerate(c.bits)) {
      if (b == RTLIL::State::S0) {
      } else if (b == RTLIL::State::S1) {
        a.setBit(idx);
      } else {
        mlir::emitError(module.getLoc())
            << " non-binary constant is not supported yet. Return random value "
               "for now";
        return builder->getIntegerAttr(builder->getIntegerType(c.size()), a);
      }
    }

    return builder->getIntegerAttr(builder->getIntegerType(c.size()), a);
  }

  Value getValueForSigSpec(const RTLIL::SigSpec &sigSpec) {
    // Wire.
    if (sigSpec.is_wire())
      return getValueForWire(sigSpec.as_wire());

    // Fully undef. Lower it to constant X.
    if (sigSpec.is_fully_undef())
      return builder->create<sv::ConstantXOp>(
          builder->getUnknownLoc(),
          builder->getIntegerType(sigSpec.as_const().size()));

    // Fully const. Lower it to constant.
    if (sigSpec.is_fully_const())
      return builder->create<hw::ConstantOp>(
          builder->getUnknownLoc(), getIntegerAttr(sigSpec.as_const()));

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
      auto chunk = sigSpec.as_chunk();
      if (!chunk.wire) {
        module.emitError() << "is not wire";
        return {};
      }
      auto v = getValueForWire(chunk.wire);
      auto width = chunk.width;
      auto offset = chunk.offset;
      return builder->create<comb::ExtractOp>(v.getLoc(), v, offset, width);
    }

    SmallVector<mlir::Value> chunks;
    for (auto w : llvm::reverse(sigSpec.chunks()))
      chunks.push_back(getValueForSigSpec(w));
    return builder->create<comb::ConcatOp>(builder->getUnknownLoc(), chunks);
  }

  // Get the value of a port of a cell.
  Value getPortValue(RTLIL::Cell *cell, StringRef portName) {
    return getValueForSigSpec(cell->getPort(getEscapedName(portName)));
  }

  SigSpec getPortSig(RTLIL::Cell *cell, StringRef portName) {
    return cell->getPort(getEscapedName(portName));
  }

  // Get the value of a port of a cell.
  Value getExtOrTruncPortValue(RTLIL::Cell *cell, StringRef portName,
                               unsigned resultWidth) {
    auto portVal = getPortValue(cell, portName);
    return getExtOrTruncPortValue(*builder, portVal, resultWidth);
  }

  LogicalResult
  importCell(const ImportRTLILDesign::ModuleMappingTy &moduleMapping,
             RTLIL::Cell *cell);

  DenseMap<StringAttr, Value> mapping;
  std::unique_ptr<OpBuilder> builder;
  llvm::MapVector<StringAttr, hw::HWModuleExternOp> exeternalModules;
  std::unique_ptr<Block> block;

  llvm::StringMap<std::unique_ptr<CellPatternBase>> handler;

  template <typename CellPattern, typename... Args>
  void addPattern(StringRef typeName, Args... args);

  template <typename OpName>
  void addOpPattern(StringRef typeName, ArrayRef<StringRef> inputPortNames,
                    StringRef outputPortName,
                    CellPatternBase::ImplicitWidthStrategy implict =
                        CellPatternBase::ImplicitWidthStrategy::None);
  template <typename OpName>
  void addOpPatternBinary(StringRef typeName,
                          CellPatternBase::ImplicitWidthStrategy implict =
                              CellPatternBase::ImplicitWidthStrategy::None);

  void registerPatterns();
};

//===----------------------------------------------------------------------===//
// Cell Patterns
//===----------------------------------------------------------------------===//

template <typename OpName>
struct CellOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    return builder.create<OpName>(location, inputValues, false);
  }
};

template <typename BinNonCommutativeOp>
struct BinNonCommutativeOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {

    return builder.create<BinNonCommutativeOp>(location, inputValues[0],
                                               inputValues[1], false);
  }
};

template <bool isAnd>
struct AndOrNotOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto notB = comb::createOrFoldNot(location, inputValues[1], builder, false);

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
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto aOrB = builder.create<OrOp>(location, inputValues, false);
    return comb::createOrFoldNot(location, aOrB, builder, false);
  }
};
struct NandOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto aOrB = builder.create<AndOp>(location, inputValues, false);
    return comb::createOrFoldNot(location, aOrB, builder, false);
  }
};

struct NotOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    return comb::createOrFoldNot(location, inputValues[0], builder, false);
  }
};

struct MuxOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    return builder.create<MuxOp>(location, inputValues[2], inputValues[1],
                                 inputValues[0]);
  }
};

struct PMuxPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto result = inputValues[0];
    auto data = inputValues[1];
    auto selector = inputValues[2];
    auto sWidth = hw::getBitWidth(selector.getType());
    auto width = hw::getBitWidth(result.getType());
    if (sWidth < 0 || width < 0)
      return {};

    // FIXME: This is inefficient.
    for (int64_t i = 0; i < sWidth; i++) {
      auto extract = builder.create<comb::ExtractOp>(location, selector, i, 1);
      auto maybeResult =
          builder.create<comb::ExtractOp>(location, data, i * width, width);
      result =
          builder.create<comb::MuxOp>(location, extract, result, maybeResult);
    }

    return result;
  }
};

struct ParityOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    return builder.create<comb::ParityOp>(location, inputValues[0]);
  }
};

template <bool isAnd>
struct ReductionOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    if (isAnd) {
      auto cnt = builder.create<hw::ConstantOp>(
          location,
          APInt::getAllOnes(hw::getBitWidth(inputValues[0].getType())));

      return builder.create<comb::ICmpOp>(location, ICmpPredicate::eq,
                                          inputValues[0], cnt);
    }
    auto zero = builder.create<hw::ConstantOp>(
        location, APInt::getZero(hw::getBitWidth(inputValues[0].getType())));
    return builder.create<comb::ICmpOp>(location, ICmpPredicate::ne,
                                        inputValues[0], zero);
  }
};

template <bool isAnd>
struct LogicAndOrPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto zero = builder.create<hw::ConstantOp>(
        location, APInt::getZero(hw::getBitWidth(inputValues[0].getType())));
    auto lhs = builder.create<comb::ICmpOp>(location, ICmpPredicate::ne,
                                            inputValues[0], zero);
    auto rhs = builder.create<comb::ICmpOp>(location, ICmpPredicate::ne,
                                            inputValues[1], zero);
    return isAnd ? builder.create<comb::AndOp>(location, lhs, rhs).getResult()
                 : builder.create<comb::OrOp>(location, lhs, rhs).getResult();
  }
};

struct NegPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto zero = builder.create<hw::ConstantOp>(
        location, APInt::getZero(hw::getBitWidth(inputValues[0].getType())));
    return builder.create<comb::SubOp>(location, zero, inputValues[0]);
  }
};

struct ICmpOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  ICmpOpPattern(StringRef typeName, ICmpPredicate pred)
      : CellPatternBase(typeName, {"A", "B"}, "Y",
                        ImplicitWidthStrategy::OperandMustBeMax),
        upred(pred), spred(pred){};

  ICmpOpPattern(StringRef typeName, ICmpPredicate pred,
                ICmpPredicate signedPred)
      : CellPatternBase(typeName, {"A", "B"}, "Y",
                        ImplicitWidthStrategy::OperandMustBeMax),
        upred(pred), spred(signedPred){};

  static FailureOr<bool> isSigned(Location location, Cell *cell) {
    bool aSigned = cell->getParam(getEscapedName("A_SIGNED")).as_bool();
    bool bSigned = cell->getParam(getEscapedName("B_SIGNED")).as_bool();
    if (aSigned && bSigned)
      return true;
    if (!aSigned && !bSigned)
      return false;
    // Currently unsupported.
    return mlir::emitError(location) << "A_SIGNED and B_SIGNED don't match";
  }

  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto singed = isSigned(location, cell);
    if (failed(singed))
      return {};

    Value lhs = inputValues[0];
    Value rhs = inputValues[1];
    return builder.create<ICmpOp>(location, *singed ? spred : upred, lhs, rhs);
  }

private:
  circt::comb::ICmpPredicate upred, spred;
};

struct XnorOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto aAndB = builder.create<AndOp>(location, inputValues, false);
    auto notA = comb::createOrFoldNot(location, inputValues[0], builder, false);
    auto notB = comb::createOrFoldNot(location, inputValues[1], builder, false);

    auto notAnds =
        builder.create<AndOp>(location, ArrayRef<Value>{notA, notB}, false);

    return builder.create<OrOp>(location, ArrayRef<Value>{aAndB, notAnds},
                                false);
  }
};

// TODO: Add async.
template <bool resetValueConst>
struct SDFFGatePattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto resetValue =
        builder.create<hw::ConstantOp>(location, APInt(1, resetValueConst));
    auto clk = builder.create<seq::ToClockOp>(location, inputValues[0]);
    return builder.create<seq::CompRegOp>(
        location, inputValues[1], clk, inputValues[2], resetValue,
        builder.getStringAttr(getStrippedName(cell->name)));
  }
};

struct DFFGatePattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto clk = builder.create<seq::ToClockOp>(location, inputValues[0]);
    auto compregOp =
        builder.create<seq::CompRegOp>(location, inputValues[1], clk);
    compregOp.setName(getStrippedName(cell->name));
    return compregOp;
  }
};

struct DFFPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto clk = builder.create<seq::ToClockOp>(location, inputValues[0]);
    return builder.create<seq::CompRegOp>(
        location, inputValues[1], clk,
        builder.getStringAttr(getStrippedName(cell->name)));
  }
};

struct DFFEPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto clk = builder.create<seq::ToClockOp>(location, inputValues[0]);
    auto en = inputValues[2];
    auto reg = builder.create<seq::CompRegOp>(
        location, inputValues[1], clk,
        builder.getStringAttr(getStrippedName(cell->name)));
    // Gate by enable.
    reg.getInputMutable().assign(
        builder.create<comb::MuxOp>(location, en, reg.getInput(), reg));
    return reg;
  }
};

struct SDFFPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto clk = builder.create<seq::ToClockOp>(location, inputValues[0]);
    auto reset = inputValues[2];
    auto resetValue = builder.create<hw::ConstantOp>(
        location, inputValues[1].getType(),
        cell->getParam(getEscapedName("SRST_VALUE")).as_int());

    return builder.create<seq::CompRegOp>(
        location, inputValues[1], clk, reset, resetValue,
        builder.getStringAttr(getStrippedName(cell->name)));
  }
};

struct SDFFEPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto clk = builder.create<seq::ToClockOp>(location, inputValues[0]);
    auto reset = inputValues[2];
    auto en = inputValues[3];
    auto resetValue = builder.create<hw::ConstantOp>(
        location, inputValues[1].getType(),
        cell->getParam(getEscapedName("SRST_VALUE")).as_int());
    auto reg = builder.create<seq::CompRegOp>(
        location, inputValues[1], clk, reset, resetValue,
        builder.getStringAttr(getStrippedName(cell->name)));

    // Gate by enable.
    reg.getInputMutable().assign(
        builder.create<comb::MuxOp>(location, en, reg.getInput(), reg));
    return reg;
  }
};

struct SDFFCEPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto clk = builder.create<seq::ToClockOp>(location, inputValues[0]);
    auto reset = inputValues[2];
    auto en = inputValues[3];
    auto resetValue = builder.create<hw::ConstantOp>(
        location, inputValues[1].getType(),
        cell->getParam(getEscapedName("SRST_VALUE")).as_int());
    reset = builder.create<comb::AndOp>(location, en, reset);
    auto reg = builder.create<seq::CompRegOp>(
        location, inputValues[1], clk, reset, resetValue,
        builder.getStringAttr(getStrippedName(cell->name)));

    // Gate by enable.
    reg.getInputMutable().assign(
        builder.create<comb::MuxOp>(location, en, reg.getInput(), reg));
    return reg;
  }
};

struct MemReadPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    auto clk = builder.create<seq::ToClockOp>(location, inputValues[0]);
    auto reset = inputValues[2];
    auto en = inputValues[3];
    auto resetValue = builder.create<hw::ConstantOp>(
        location, inputValues[1].getType(),
        cell->getParam(getEscapedName("SRST_VALUE")).as_int());
    auto reg = builder.create<seq::CompRegOp>(
        location, inputValues[1], clk, reset, resetValue,
        builder.getStringAttr(getStrippedName(cell->name)));

    // Gate by enable.
    reg.getInputMutable().assign(
        builder.create<comb::MuxOp>(location, en, reg.getInput(), reg));
    return reg;
  }
};

} // namespace

template <typename CellPattern, typename... Args>
void ImportRTLILModule::addPattern(StringRef typeName, Args... args) {
  handler.insert({typeName, std::make_unique<CellPattern>(typeName, args...)});
}

template <typename OpName>
void ImportRTLILModule::addOpPattern(
    StringRef typeName, ArrayRef<StringRef> inputPortNames,
    StringRef outputPortName, CellPatternBase::ImplicitWidthStrategy st) {
  handler.insert({typeName, std::make_unique<CellOpPattern<OpName>>(
                                typeName, inputPortNames, outputPortName)});
}

template <typename OpName>
void ImportRTLILModule::addOpPatternBinary(
    StringRef typeName, CellPatternBase::ImplicitWidthStrategy implicit) {
  handler.insert(
      {typeName, std::make_unique<CellOpPattern<OpName>>(
                     typeName, ArrayRef<StringRef>{"A", "B"}, "Y", implicit)});
}

ImportRTLILModule::ImportRTLILModule(MLIRContext *context,
                                     const ImportRTLILDesign &importer,
                                     RTLIL::Module *rtlilModule,
                                     OpBuilder &moduleBuilder)
    : rtlilModule(rtlilModule) {
  builder = std::make_unique<OpBuilder>(context);
  block = std::make_unique<Block>();
  registerPatterns();
  size_t size = rtlilModule->ports.size();
  SmallVector<hw::PortInfo> ports(size);
  SmallVector<Value> values(size);
  auto modName =  getStringAttr( rtlilModule->name);

  size_t numInput = 0, numOutput = 0;
  for (const auto &port : rtlilModule->ports) {
    auto *wire = rtlilModule->wires_[port];
    assert(wire->port_input || wire->port_output);

    // Port id is 1-indexed.
    size_t portId = wire->port_id - 1;
    size_t argNum = (wire->port_output ? numOutput : numInput)++;
    ports[portId].name = getStringAttr(wire->name);
    ports[portId].argNum = argNum;
    ports[portId].type = builder->getIntegerType(wire->width);
    ports[portId].dir =
        wire->port_output ? hw::ModulePort::Output : hw::ModulePort::Input;
  }
  SmallVector<Attribute> defaultParameters;
  for (auto [paramName, param] : rtlilModule->parameter_default_values) {
    auto value =
        IntegerAttr::get(builder->getIntegerType(param.size()), param.as_int());
    auto type = builder->getIntegerType(param.size());
    auto name = getStringAttr(paramName);
    defaultParameters.push_back(
        builder->getAttr<hw::ParamDeclAttr>(name, type, value));
  }
  auto paramArgs = builder->getArrayAttr(defaultParameters);

  if (rtlilModule->get_blackbox_attribute()) {
    module = moduleBuilder.create<hw::HWModuleExternOp>(
        UnknownLoc::get(context), modName, ports, modName, paramArgs);
    module.setPrivate();
  } else
    module = moduleBuilder.create<hw::HWModuleOp>(UnknownLoc::get(context),
                                                  modName, ports, paramArgs);
}

mlir::TypedValue<hw::InOutType>
ImportRTLILModule::getInOutValue(Location loc, const SigSpec &sigSpec) {
  if (sigSpec.is_wire())
    return getWireValue(sigSpec.as_wire());

  // Const cannot be inout.
  if (sigSpec.is_fully_const())
    return {};

  // Bit selection.
  if (sigSpec.is_bit()) {
    auto bit = sigSpec.as_bit();
    if (!bit.wire) {
      module.emitError() << "is not wire";
      return {};
    }
    auto wire = getWireValue(bit.wire);
    assert(wire);
    auto idx = builder->create<hw::ConstantOp>(
        wire.getLoc(),
        APInt(bit.wire->width <= 1 ? 1 : llvm::Log2_32_Ceil(bit.wire->width),
              bit.offset));
    return builder->create<sv::ArrayIndexInOutOp>(wire.getLoc(), wire, idx);
  }

  // Range selection.
  if (sigSpec.is_chunk()) {
    auto chunk = sigSpec.as_chunk();
    if (!chunk.wire) {
      mlir::emitError(loc) << "unsupported chunk states";
      return {};
    }

    auto wire = getWireValue(chunk.wire);
    auto arrayLength =
        cast<hw::ArrayType>(wire.getElementType()).getNumElements();
    auto idx = builder->create<hw::ConstantOp>(
        wire.getLoc(),
        APInt(arrayLength <= 1 ? 1 : llvm::Log2_32_Ceil(arrayLength),
              chunk.offset));
    return builder->create<sv::IndexedPartSelectInOutOp>(loc, wire, idx,
                                                         chunk.width);
  }

  // Concat ref.
  auto size = sigSpec.size();
  auto newWire = builder->create<sv::WireOp>(
      loc, hw::ArrayType::get(builder->getI1Type(), size));
  size_t newOffest = 0;
  for (auto sig : sigSpec.chunks()) {
    if (!sig.is_wire()) {
      mlir::emitError(loc) << "unsupported chunk";
      return {};
    }
    auto child = wireMapping.lookup(getStringAttr(sig.wire));
    auto childSize =
        cast<hw::ArrayType>(child.getElementType()).getNumElements();
    auto width = sig.width;
    auto offset = sig.offset;
    auto idx = builder->create<hw::ConstantOp>(
        loc, APInt(childSize <= 1 ? 1 : llvm::Log2_32_Ceil(childSize), offset));
    auto parent =
        builder->create<sv::IndexedPartSelectInOutOp>(loc, child, idx, width);

    auto newIndex = builder->create<hw::ConstantOp>(
        loc, APInt(size <= 1 ? 1 : llvm::Log2_32_Ceil(size), newOffest));
    auto newRhs = builder->create<sv::IndexedPartSelectInOutOp>(
        loc, newWire, newIndex, width);

    // Make sure offset is correct.
    // parent <= wire[t, offset]
    builder->create<sv::AssignOp>(
        loc, parent, builder->create<sv::ReadInOutOp>(loc, newRhs));
  }

  return newWire;
}

LogicalResult ImportRTLILModule::connect(Location loc,
                                         mlir::TypedValue<hw::InOutType> lhs,
                                         Value rhs) {
  if (lhs.getType().getElementType() != rhs.getType())
    rhs = builder->create<hw::BitcastOp>(loc, lhs.getType().getElementType(),
                                         rhs);
  builder->create<sv::AssignOp>(loc, lhs, rhs);
  return success();
}

LogicalResult ImportRTLILModule::connect(Location loc, const SigSpec &lhs,
                                         const SigSpec &rhs) {
  auto lhsValue = getInOutValue(loc, lhs);
  Value output = getValueForSigSpec(rhs);
  if (!lhsValue || !output)
    return mlir::emitError(loc) << "unsupported connection";

  return connect(loc, lhsValue, output);
}

LogicalResult ImportRTLILModule::importCell(
    const ImportRTLILDesign::ModuleMappingTy &moduleMapping,
    RTLIL::Cell *cell) {
  auto cellName = getStringAttr(cell->type);
  LLVM_DEBUG(llvm::dbgs() << "Importing Cell " << cellName << "\n";);
  // Standard cells.
  auto it = handler.find(cellName);
  auto location = builder->getUnknownLoc();

  if (it == handler.end()) {

    DenseMap<StringAttr, mlir::TypedValue<hw::InOutType>> lhsMap;
    SmallVector<Value> values;
    SmallVector<std::pair<int, Value>> rhsValuesWithIndex;
    SmallVector<hw::PortInfo> ports;
    HWModuleLike referredModule;
    auto *referredRTLILModule = rtlilModule->design->module(cell->type);
    for (auto [lhs, rhs] : cell->connections()) {
      hw::PortInfo hwPort;
      hwPort.name = getStringAttr(lhs);
      auto portIdx =
          referredRTLILModule ? referredRTLILModule->wire(lhs)->port_id : 0;

      if (cell->output(lhs)) {
        hwPort.dir = hw::PortInfo::Output;
        auto value = getInOutValue(location, rhs);
        if (!value)
          return mlir::emitError(location)
                 << "port lowering failed cell name=" << cellName

                 << " port name=" << lhs.c_str();
        lhsMap[hwPort.name] = value;

        auto array = dyn_cast<hw::ArrayType>(value.getType().getElementType());
        hwPort.type =
            builder->getIntegerType(array ? array.getNumElements() : 1);
      } else {
        hwPort.dir = hw::PortInfo::Input;
        rhsValuesWithIndex.push_back({portIdx, getValueForSigSpec(rhs)});

        if (!rhsValuesWithIndex.back().second)
          return mlir::emitError(location)
                 << "port lowering failed cell name=" << cellName
                 << " port name=" << lhs.c_str();

        hwPort.type = rhsValuesWithIndex.back().second.getType();
      }

      ports.push_back(hwPort);
    }
    ArrayAttr paramDecl, paramArgs;
    if (cell->parameters.size()) {
      SmallVector<Attribute> parameters, defaultParameters;
      for (auto [paramName, param] : cell->parameters) {
        auto value = IntegerAttr::get(builder->getIntegerType(param.size()),
                                      param.as_int());
        auto type = builder->getIntegerType(param.size());
        auto name = getStringAttr(paramName);
        parameters.push_back(
            builder->getAttr<hw::ParamDeclAttr>(name, type, Attribute()));
        defaultParameters.push_back(
            builder->getAttr<hw::ParamDeclAttr>(name, type, value));
      }
      paramDecl = builder->getArrayAttr(parameters);
      paramArgs = builder->getArrayAttr(defaultParameters);
    }

    if (cellName.getValue().starts_with("$")) {
      // Yosys std cells. Just lower it to external module instances for now.
      auto &extMod = exeternalModules[cellName];
      if (!extMod)
        return mlir::emitError(location)
               << "Found unknown builtin cell " << cellName;

      referredModule = extMod;
    } else {
      // Otherwise lower it to an instance.
      auto mod = moduleMapping.lookup(cellName);
      if (!mod)
        return failure();
      referredModule = mod;
    }

    if (referredRTLILModule)
      std::sort(rhsValuesWithIndex.begin(), rhsValuesWithIndex.end(),
                [](const auto &lhs, const auto &rhs) {
                  return lhs.first < rhs.first;
                });

    for (auto t : rhsValuesWithIndex)
      values.push_back(t.second);

    auto result = builder->create<hw::InstanceOp>(
        location, referredModule, getStringAttr(cell->name), values, paramArgs);
    for (auto [idx, rhs] : llvm::enumerate(referredModule.getOutputNames())) {
      auto it = lhsMap.find(cast<StringAttr>(rhs));
      if (it == lhsMap.end())
        continue;
      if (failed(connect(location, it->second, result.getResult(idx))))
        return failure();
    }
    return success();
  }
  auto result = it->second->convert(*this, cell);
  return result;
}

// RTLIL importer creates a bunch of array wires to mimic bit-sensitive
// assignments. Eliminate all of them in the post-process.
struct Range {
  Range() = default;
  sv::WireOp root = {};
  int64_t start = 0;
  int64_t width = 0;
};

static std::optional<int64_t> getConstantValue(Value value) {
  auto constantIndex = value.template getDefiningOp<hw::ConstantOp>();
  if (constantIndex)
    if (constantIndex.getValue().getBitWidth() <= 63)
      return constantIndex.getValue().getZExtValue();

  return {};
}

static Range getRoot(Value value) {
  auto op = value.getDefiningOp();
  if (!op)
    return Range();
  auto width =
      hw::getBitWidth(cast<hw::InOutType>(value.getType()).getElementType());
  return TypeSwitch<Operation *, Range>(op)
      .Case<sv::WireOp>([&](auto op) { return Range{op, 0, width}; })
      .Case<sv::ArrayIndexInOutOp>([&](sv::ArrayIndexInOutOp op) {
        auto result = getRoot(op.getInput());
        auto index = getConstantValue(op.getIndex());
        if (!index)
          return Range();
        result.start += *index;
        result.width = width;
        return result;
      })
      .Case<sv::IndexedPartSelectInOutOp>([&](sv::IndexedPartSelectInOutOp op) {
        auto result = getRoot(op.getInput());
        auto index = getConstantValue(op.getBase());
        if (!index)
          return Range();
        result.start += *index;
        result.width = width;
        return result;
      })
      .Default([](auto) { return Range(); });
}
// Eliminate all SV wires used in the module.
static LogicalResult cleanUpHWModule(hw::HWModuleOp module) {
  llvm::MapVector<sv::WireOp, SmallVector<std::pair<Range, sv::AssignOp>>>
      writes;
  llvm::MapVector<sv::WireOp, SmallVector<std::pair<Range, Value>>> reads;
  module.walk([&](sv::WireOp wire) { writes.insert({wire, {}}); });
  module.walk([&](sv::AssignOp assign) {
    auto range = getRoot(assign.getDest());
    if (range.root)
      writes[range.root].emplace_back(range, assign);
  });
  module.walk([&](sv::ReadInOutOp read) {
    auto range = getRoot(read.getInput());
    if (range.root)
      reads[range.root].emplace_back(range, read);
  });
  for (auto &[wire, elements] : writes) {
    // Sort by start index.
    llvm::sort(elements.begin(), elements.end(),
               [](const auto &lhs, const auto &rhs) {
                 return lhs.first.start < rhs.first.start;
               });
    int64_t currentIdx = 0;
    bool failed = false;
    for (auto elem : elements) {
      if (elem.first.start != currentIdx || elem.first.width < 0) {
        failed = true;
        break;
      }
      currentIdx += elem.first.width;
    }

    if (failed)
      continue;

    // Fully written.
    if (currentIdx == hw::getBitWidth(wire.getType().getElementType())) {
      OpBuilder builder(wire);
      SmallVector<Value> concatInputs;
      for (auto elem : elements) {
        // Bitcast to integer and comb concat.
        auto bitcast = builder.createOrFold<hw::BitcastOp>(
            wire.getLoc(), builder.getIntegerType(elem.first.width),
            elem.second.getSrc());
        concatInputs.push_back(bitcast);
        elem.second->erase();
      }
      std::reverse(concatInputs.begin(), concatInputs.end());
      auto concat =
          builder.createOrFold<comb::ConcatOp>(wire.getLoc(), concatInputs);

      for (auto &[range, readOp] : reads[wire]) {
        auto extract = builder.create<comb::ExtractOp>(
            wire.getLoc(), concat, range.start, range.width);
        readOp.replaceAllUsesWith(extract);
        readOp.getDefiningOp()->erase();
      }
    }
  }

  return success();
}
LogicalResult ImportRTLILModule::importBody(
    const ImportRTLILDesign::ModuleMappingTy &moduleMapping) {
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
      mapping.insert({getStringAttr(wire->name), arg});
    } else {
      auto loc = builder->getUnknownLoc();
      auto w = builder->create<sv::WireOp>(
          loc, hw::ArrayType::get(builder->getIntegerType(1), wire->width));
      auto read = builder->create<hw::BitcastOp>(
          loc, builder->getIntegerType(wire->width),
          builder->create<sv::ReadInOutOp>(loc, w));
      mapping.insert({getStringAttr(wire), read});
      wireMapping.insert({getStringAttr(wire), w});
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

  // Clean up the module by removing SV constructs.
  if (auto hwModule = dyn_cast<hw::HWModuleOp>(*module))
    if (failed(cleanUpHWModule(hwModule)))
      return failure();

  return success();
}

LogicalResult CellPatternBase::convert(ImportRTLILModule &importer,
                                       Cell *cell) {
  // TODO: Translate location.
  auto location = importer.builder->getUnknownLoc();

  auto lhsSig = importer.getPortSig(cell, outputPortName);
  auto lhsValue = importer.getInOutValue(location, lhsSig);
  int64_t resultWidth = -1;
  if (implicitWidthStrategy == HasOperandsAndResultSameWidth) {
    resultWidth = hw::getBitWidth(lhsValue.getType().getElementType());
  }

  SmallVector<Value> inputs;
  for (const auto &name : inputPortNames) {
    auto portVal = importer.getPortValue(cell, name);
    if (!portVal)
      return failure();
    if (implicitWidthStrategy == HasOperandsAndResultSameWidth)
      portVal = importer.getExtOrTruncPortValue(*importer.builder, portVal,
                                                resultWidth);

    inputs.push_back(portVal);
  }

  if (implicitWidthStrategy == OperandMustBeMax) {
    unsigned int width = 0;
    for (auto input : inputs) {
      width = std::max(width, input.getType().getIntOrFloatBitWidth());
    }
    for (size_t i = 0; i < inputs.size(); i++)
      inputs[i] = ImportRTLILModule::getExtOrTruncPortValue(*importer.builder,
                                                            inputs[i], width);
  }

  auto rhsValue = convert(cell, *importer.builder, location, inputs);
  if (!rhsValue)
    return failure();

  if (auto *op = rhsValue.getDefiningOp())
    if (failed(mlir::verify(op, false))) {
      mlir::emitError(location)
          << getStrippedName(cell->name) << " lowering failed";
      op->dump();
      return failure();
    }

  return importer.connect(location, lhsValue, rhsValue);
}

LogicalResult ImportRTLILDesign::run(mlir::ModuleOp module) {
  SmallVector<ImportRTLILModule> modules;
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
    for (auto [str, ext] : mod.getExternalModules()) {
      auto it = extMap.insert(str).second;
      if (it) {
        ext->moveBefore(module.getBody(), module.getBody()->begin());
      }
    }
  }

  return success();
}

LogicalResult circt::rtlil::importRTLILDesign(RTLIL::Design *design,
                                              ModuleOp module) {
  ImportRTLILDesign importer(design);
  return importer.run(module);
}

void ImportRTLILModule::registerPatterns() {
  // Comb primitive cells.
  addOpPatternBinary<comb::AddOp>(
      "$add",
      CellPatternBase::ImplicitWidthStrategy::HasOperandsAndResultSameWidth);
  addOpPatternBinary<comb::AndOp>(
      "$and",
      CellPatternBase::ImplicitWidthStrategy::HasOperandsAndResultSameWidth);
  addOpPatternBinary<comb::OrOp>(
      "$or",
      CellPatternBase::ImplicitWidthStrategy::HasOperandsAndResultSameWidth);
  addOpPatternBinary<comb::XorOp>(
      "$xor",
      CellPatternBase::ImplicitWidthStrategy::HasOperandsAndResultSameWidth);
  addOpPatternBinary<comb::MulOp>(
      "$mul",
      CellPatternBase::ImplicitWidthStrategy::HasOperandsAndResultSameWidth);
  addPattern<BinNonCommutativeOpPattern<SubOp>>(
      "$sub", ArrayRef<StringRef>{"A", "B"}, "Y",
      CellPatternBase::ImplicitWidthStrategy::HasOperandsAndResultSameWidth);
  addPattern<BinNonCommutativeOpPattern<ShlOp>>(
      "$shl", ArrayRef<StringRef>{"A", "B"}, "Y",
      CellPatternBase::ImplicitWidthStrategy::HasOperandsAndResultSameWidth);
  addPattern<BinNonCommutativeOpPattern<ShrSOp>>(
      "$sshr", ArrayRef<StringRef>{"A", "B"}, "Y",
      CellPatternBase::ImplicitWidthStrategy::HasOperandsAndResultSameWidth);

  addPattern<MuxOpPattern>("$mux", ArrayRef<StringRef>{"A", "B", "S"}, "Y");
  addPattern<ParityOpPattern>(
      "$reduce_xor", ArrayRef<StringRef>{"A"}, "Y",
      CellPatternBase::ImplicitWidthStrategy::OperandMustBeMax);
  addPattern<ReductionOpPattern<true>>(
      "$reduce_and", ArrayRef<StringRef>{"A"}, "Y",
      CellPatternBase::ImplicitWidthStrategy::OperandMustBeMax);
  addPattern<ReductionOpPattern<false>>(
      "$reduce_bool", ArrayRef<StringRef>{"A"}, "Y",
      CellPatternBase::ImplicitWidthStrategy::OperandMustBeMax);
  addPattern<ReductionOpPattern<false>>(
      "$reduce_or", ArrayRef<StringRef>{"A"}, "Y",
      CellPatternBase::ImplicitWidthStrategy::OperandMustBeMax);
  addPattern<ReductionOpPattern<false>>(
      "$logic_not", ArrayRef<StringRef>{"A"}, "Y",
      CellPatternBase::ImplicitWidthStrategy::OperandMustBeMax);
  addPattern<LogicAndOrPattern<true>>(
      "$logic_and", ArrayRef<StringRef>{"A", "B"}, "Y",
      CellPatternBase::ImplicitWidthStrategy::OperandMustBeMax);
  addPattern<LogicAndOrPattern<false>>(
      "$logic_or", ArrayRef<StringRef>{"A", "B"}, "Y",
      CellPatternBase::ImplicitWidthStrategy::OperandMustBeMax);
  addPattern<NegPattern>("$neg", ArrayRef<StringRef>{"A"}, "Y");

  addPattern<NotOpPattern>("$not", ArrayRef<StringRef>{"A"}, "Y");
  addPattern<PMuxPattern>("$pmux", ArrayRef<StringRef>{"A", "B", "S"}, "Y");

  // ICmp.
  addPattern<ICmpOpPattern>("$eq", ICmpPredicate::eq);
  addPattern<ICmpOpPattern>("$ne", ICmpPredicate::ne);
  addPattern<ICmpOpPattern>("$le", ICmpPredicate::ule, ICmpPredicate::sle);
  addPattern<ICmpOpPattern>("$lt", ICmpPredicate::ult, ICmpPredicate::slt);
  addPattern<ICmpOpPattern>("$ge", ICmpPredicate::uge, ICmpPredicate::sge);
  addPattern<ICmpOpPattern>("$gt", ICmpPredicate::ugt, ICmpPredicate::sgt);

  // FIXME: Check the CLK_POLARITY attribute.
  addPattern<DFFPattern>("$dff", ArrayRef<StringRef>{"CLK", "D"}, "Q");
  addPattern<DFFEPattern>("$dffe", ArrayRef<StringRef>{"CLK", "D", "EN"}, "Q");
  addPattern<SDFFPattern>("$sdff", ArrayRef<StringRef>{"CLK", "D", "SRST"},
                          "Q");
  addPattern<SDFFEPattern>("$sdffe",
                           ArrayRef<StringRef>{"CLK", "D", "SRST", "EN"}, "Q");
  addPattern<SDFFCEPattern>("$sdffce",
                            ArrayRef<StringRef>{"CLK", "D", "SRST", "EN"}, "Q");

  // Post-synthesis gate cells.
  addOpPatternBinary<comb::XorOp>("$_XOR_");
  addOpPatternBinary<comb::AndOp>("$_AND_");
  addOpPatternBinary<comb::OrOp>("$_OR_");
  addPattern<AndOrNotOpPattern</*isAnd=*/true>>(
      "$_ANDNOT_", ArrayRef<StringRef>{"A", "B"}, "Y");
  addPattern<AndOrNotOpPattern</*isAnd=*/false>>(
      "$_ORNOT_", ArrayRef<StringRef>{"A", "B"}, "Y");
  addPattern<XnorOpPattern>("$_XNOR_", ArrayRef<StringRef>{"A", "B"}, "Y");
  addPattern<NorOpPattern>("$_NOR_", ArrayRef<StringRef>{"A", "B"}, "Y");
  addPattern<NandOpPattern>("$_NAND_", ArrayRef<StringRef>{"A", "B"}, "Y");
  addPattern<NotOpPattern>("$_NOT_", ArrayRef<StringRef>{"A"}, "Y");
  addPattern<MuxOpPattern>("$_MUX_", ArrayRef<StringRef>{"A", "B", "S"}, "Y");
  addPattern<SDFFGatePattern</*resetValueConst=*/false>>(
      "$_SDFF_PP0_", ArrayRef<StringRef>{"C", "D", "R"}, "Q");
  addPattern<SDFFGatePattern</*resetValueConst=*/true>>(
      "$_SDFF_PP1_", ArrayRef<StringRef>{"C", "D", "R"}, "Q");
  addPattern<DFFGatePattern>("$_DFF_P_", ArrayRef<StringRef>{"C", "D"}, "Q");
}

void circt::rtlil::registerRTLILImport() {
  static mlir::TranslateToMLIRRegistration fromRTLIL(
      "import-rtlil", "import RTLIL",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        context->loadDialect<hw::HWDialect, comb::CombDialect, sv::SVDialect,
                             seq::SeqDialect>();

        OwningOpRef<ModuleOp> module(
            ModuleOp::create(UnknownLoc::get(context)));
        if (sourceMgr.getNumBuffers() != 1) {
          module = {};
          return module;
        }
        std::error_code ec;
        SmallString<128> fileName;
        if ((ec = llvm::sys::fs::createTemporaryFile("yosys", "rtlil",
                                                     fileName))) {
          module->emitError() << ec.message();
          module = {};
          return module;
        }

        StringRef ref =
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer();
        std::ofstream myfile(fileName.c_str());
        myfile.write(ref.data(), ref.size());
        myfile.close();

        init_yosys(false);

        auto design = std::make_unique<RTLIL::Design>();
        std::string cmd = "read_rtlil ";
        cmd += fileName;
        Yosys::run_pass(cmd, design.get());

        if (failed(importRTLILDesign(design.get(), module.get()))) {
          mlir::emitError(module->getLoc()) << "import failed";
          module = {};
        }
        return module;
      });
}