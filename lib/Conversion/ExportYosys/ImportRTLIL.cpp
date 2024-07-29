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
#include "RTLILConverterInternal.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

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

namespace {
class ImportRTLILModule;
class CellPatternBase {
public:
  CellPatternBase(StringRef typeName, ArrayRef<StringRef> inputPortNames,
                  StringRef outputPortName)
      : typeName(typeName), inputPortNames(inputPortNames),
        outputPortName(outputPortName){};
  LogicalResult convert(ImportRTLILModule &importer, Cell *cell);
  virtual ~CellPatternBase() {}

private:
  virtual Value convert(Cell *cell, OpBuilder &builder, Location location,
                        ValueRange inputValues) = 0;

  SmallString<4> typeName;
  SmallVector<SmallString<4>> inputPortNames;
  SmallString<4> outputPortName;
};
struct ImportRTLILDesign {
  RTLIL::Design *design;
  ImportRTLILDesign(RTLIL::Design *design) : design(design) {}
  LogicalResult run(mlir::ModuleOp module);
  using ModuleMappingTy = DenseMap<StringAttr, hw::HWModuleLike>;
  ModuleMappingTy moduleMapping;
};

class ImportRTLILModule {
public:
  ImportRTLILModule(MLIRContext *context, const ImportRTLILDesign &importer,
                    RTLIL::Module *rtlilModule, OpBuilder &moduleBuilder);

  hw::HWModuleLike getModuleOp() { return module; }
  LogicalResult
  importBody(const ImportRTLILDesign::ModuleMappingTy &moduleMapping);
  const auto &getExternalModules() const { return exeternalModules; }

  friend class CellPatternBase;

private:
  RTLIL::Module *rtlilModule;
  hw::HWModuleLike module;
  MLIRContext *context;
  const ImportRTLILDesign &importer;

  StringAttr getStringAttr(const Yosys::RTLIL::IdString &str) const {
    StringRef s(str.c_str());
    return builder->getStringAttr(s.starts_with("\\") ? s.drop_front(1) : s);
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
            << " non-binary constant is not supported yet";
        return {};
      }
      return builder->getIntegerAttr(builder->getIntegerType(c.size()), a);
    }

    return builder->getIntegerAttr(builder->getIntegerType(c.size()),
                                   APInt(c.size(), c.as_int()));
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

  Value getPortValue(RTLIL::Cell *cell, StringRef portName) {
    return getValueForSigSpec(cell->getPort(getEscapedName(portName)));
  }
  SigSpec getPortSig(RTLIL::Cell *cell, StringRef portName) {
    return cell->getPort(getEscapedName(portName));
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
                    StringRef outputPortName);
  template <typename OpName>
  void addOpPatternBinary(StringRef typeName);

  void registerPatterns();
};

template <typename OpName>
struct CellOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    return builder.create<OpName>(location, inputValues, false);
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

struct ICmpOpPattern : public CellPatternBase {
  using CellPatternBase::CellPatternBase;
  ICmpOpPattern(StringRef typeName, ICmpPredicate pred)
      : CellPatternBase(typeName, {"A", "B"}, "Y"), pred(pred){};

  ICmpOpPattern(StringRef typeName, ICmpPredicate pred,
                ICmpPredicate signedPred)
      : CellPatternBase(typeName, {"A", "B"}, "Y"), pred(pred){};

  Value convert(Cell *cell, OpBuilder &builder, Location location,
                ValueRange inputValues) override {
    return builder.create<ICmpOp>(location, pred, inputValues[0],
                                  inputValues[1]);
  }

private:
  circt::comb::ICmpPredicate pred;
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

} // namespace

template <typename CellPattern, typename... Args>
void ImportRTLILModule::addPattern(StringRef typeName, Args... args) {
  handler.insert({typeName, std::make_unique<CellPattern>(typeName, args...)});
}

template <typename OpName>
void ImportRTLILModule::addOpPattern(StringRef typeName,
                                     ArrayRef<StringRef> inputPortNames,
                                     StringRef outputPortName) {
  handler.insert({typeName, std::make_unique<CellOpPattern<OpName>>(
                                typeName, inputPortNames, outputPortName)});
}

template <typename OpName>
void ImportRTLILModule::addOpPatternBinary(StringRef typeName) {
  handler.insert({typeName, std::make_unique<CellOpPattern<OpName>>(
                                typeName, ArrayRef<StringRef>{"A", "B"}, "Y")});
}

void ImportRTLILModule::registerPatterns() {
  // Comb primitive cells.
  addOpPatternBinary<comb::AddOp>("$add");
  addOpPatternBinary<comb::AndOp>("$and");
  addOpPatternBinary<comb::OrOp>("$or");
  addOpPatternBinary<comb::XorOp>("$xor");
  addOpPatternBinary<comb::MulOp>("$mul");
  addPattern<MuxOpPattern>("$mux", ArrayRef<StringRef>{"A", "B", "S"}, "Y");
  addPattern<ICmpOpPattern>("$eq", ICmpPredicate::eq);
  addPattern<ICmpOpPattern>("$ne", ICmpPredicate::ne);
  addPattern<ICmpOpPattern>("$lt", ICmpPredicate::ult);

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
}

ImportRTLILModule::ImportRTLILModule(MLIRContext *context,
                                     const ImportRTLILDesign &importer,
                                     RTLIL::Module *rtlilModule,
                                     OpBuilder &moduleBuilder)
    : rtlilModule(rtlilModule), context(context), importer(importer) {
  builder = std::make_unique<OpBuilder>(context);
  block = std::make_unique<Block>();
  registerPatterns();
  size_t size = rtlilModule->ports.size();
  SmallVector<hw::PortInfo> ports(size);
  SmallVector<Value> values(size);
  auto modName = getStringAttr(rtlilModule->name);

  size_t numInput = 0, numOutput = 0;
  for (const auto &port : rtlilModule->ports) {
    auto *wire = rtlilModule->wires_[port];
    assert(wire->port_input || wire->port_output);

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

    SmallVector<mlir::TypedValue<hw::InOutType>> lhsValues;
    SmallVector<std::pair<int, mlir::TypedValue<hw::InOutType>>>
        lhsValuesWithIndex;
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
        lhsValuesWithIndex.push_back({portIdx, getInOutValue(location, rhs)});
        if (!lhsValuesWithIndex.back().second)
          return mlir::emitError(location)
                 << "port lowering failed cell name=" << cellName

                 << " port name=" << lhs.c_str();
        auto array = dyn_cast<hw::ArrayType>(
            lhsValuesWithIndex.back().second.getType().getElementType());
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
      // Yosys std cells. Just lower it to external module instances.
      auto &extMod = exeternalModules[cellName];
      if (!extMod) {
        OpBuilder::InsertionGuard guard(*builder);
        builder->setInsertionPointToStart(block.get());
        SmallString<16> name;
        name += "yosys_builtin_cell";
        name += cellName;
        extMod = builder->create<hw::HWModuleExternOp>(location, cellName,
                                                       ports, name, paramDecl);
        extMod.setPrivate();
      }
      referredModule = extMod;
    } else {
      // Otherwise lower it to an instance.
      auto mod = moduleMapping.lookup(cellName);
      if (!mod)
        return failure();
      referredModule = mod;
    }

    if (referredRTLILModule) {
      std::sort(lhsValuesWithIndex.begin(), lhsValuesWithIndex.end(),
                [](const auto &lhs, const auto &rhs) {
                  return lhs.first < rhs.first;
                });
      std::sort(rhsValuesWithIndex.begin(), rhsValuesWithIndex.end(),
                [](const auto &lhs, const auto &rhs) {
                  return lhs.first < rhs.first;
                });
    }
    for (auto t : lhsValuesWithIndex)
      lhsValues.push_back(t.second);
    for (auto t : rhsValuesWithIndex)
      values.push_back(t.second);

    auto result = builder->create<hw::InstanceOp>(
        location, referredModule, getStringAttr(cell->name), values, paramArgs);
    assert(result.getNumResults() == lhsValues.size());
    for (auto [lhs, rhs] : llvm::zip(lhsValues, result.getResults()))
      if (failed(connect(location, lhs, (Value)rhs)))
        return failure();
    return success();
  }
  auto result = it->second->convert(*this, cell);
  return result;
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

  return success();
}

LogicalResult CellPatternBase::convert(ImportRTLILModule &importer,
                                       Cell *cell) {
  SmallVector<Value> inputs;
  for (auto name : inputPortNames) {
    inputs.push_back(importer.getPortValue(cell, name));
    if (!inputs.back())
      return failure();
  }
  auto location = importer.builder->getUnknownLoc();

  auto rhsValue = convert(cell, *importer.builder, location, inputs);
  auto lhsSig = importer.getPortSig(cell, outputPortName);
  auto lhsValue = importer.getInOutValue(location, lhsSig);
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
