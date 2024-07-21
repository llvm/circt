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
#include "../PassDetail.h"
#include "RTLILConverterInternal.h"
#include "circt/Conversion/ExportRTLIL.h"
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

struct ImportRTLILDesign {
  RTLIL::Design *design;
  LogicalResult run(mlir::ModuleOp module);
  using ModuleMappingTy = DenseMap<StringAttr, hw::HWModuleLike>;
  ModuleMappingTy moduleMapping;
};

struct ImportRTLILModule {
  RTLIL::Module *rtlilModule;
  hw::HWModuleLike module;
  MLIRContext *context;
  hw::HWModuleLike getModuleOp() { return module; }
  ImportRTLILModule(MLIRContext *context, const ImportRTLILDesign &importer,
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
  const ImportRTLILDesign &importer;
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
          APInt(bit.wire->width <= 1 ? 1 : llvm::Log2_32_Ceil(bit.wire->width),
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
          APInt(arrayLength <= 1 ? 1 : llvm::Log2_32_Ceil(arrayLength),
                chunk.offset));
      return builder->create<sv::IndexedPartSelectInOutOp>(loc, v, idx,
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
      auto child = wireMapping.lookup(getStr(sig.wire->name));
      auto childSize =
          child.getElementType().cast<hw::ArrayType>().getNumElements();
      auto width = sig.width;
      auto offset = sig.offset;
      auto idx = builder->create<hw::ConstantOp>(
          loc,
          APInt(childSize <= 1 ? 1 : llvm::Log2_32_Ceil(childSize), offset));
      auto parent =
          builder->create<sv::IndexedPartSelectInOutOp>(loc, child, idx, width);

      auto newIndex = builder->create<hw::ConstantOp>(
          loc, APInt(size <= 1 ? 1 : llvm::Log2_32_Ceil(size), newOffest));
      auto newRhs = builder->create<sv::IndexedPartSelectInOutOp>(loc, newWire,
                                                                  idx, width);

      // Make sure offset is correct.
      // parent <= wire[t, offset]
      builder->create<sv::AssignOp>(
          loc, parent, builder->create<sv::ReadInOutOp>(loc, newRhs));
    }

    return newWire;

    // mlir::emitError(loc) << "unsupported lhs value";
    // return {};
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
  importBody(const ImportRTLILDesign::ModuleMappingTy &moduleMapping) {
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
      if (sigSpec.as_const().size() > 32) {
        APInt a = APInt::getZero(sigSpec.as_const().size());
        for (auto [idx, b] : llvm::enumerate(sigSpec.as_const().bits)) {
          if (b == RTLIL::State::S0) {
          } else if (b == RTLIL::State::S1) {
            a.setBit(idx);
          } else {
            mlir::emitError(module.getLoc())
                << " non-binary constant is not supported yet";
            return Value();
          }
        }

        return builder->create<hw::ConstantOp>(builder->getUnknownLoc(), a);
      }
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
    LogicalResult convert(ImportRTLILModule &importer, Cell *cell) {
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

  LogicalResult
  importCell(const ImportRTLILDesign::ModuleMappingTy &moduleMapping,
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
      SmallVector<std::pair<int, mlir::TypedValue<hw::InOutType>>>
          lhsValuesWithIndex;
      SmallVector<Value> values;
      SmallVector<std::pair<int, Value>> rhsValuesWithIndex;
      SmallVector<hw::PortInfo> ports;
      Operation *referredModule;
      auto *referredRTLILModule = rtlilModule->design->module(cell->type);
      for (auto [lhs, rhs] : cell->connections()) {
        hw::PortInfo hwPort;
        hwPort.name = getStr(lhs);
        auto portIdx =
            referredRTLILModule ? referredRTLILModule->wire(lhs)->port_id : 0;

        if (cell->output(lhs)) {
          hwPort.dir = hw::PortInfo::Output;
          lhsValuesWithIndex.push_back({portIdx, getInOutValue(location, rhs)});
          if (!lhsValuesWithIndex.back().second)
            return mlir::emitError(location)
                   << "port lowering failed cell name=" << v

                   << " port name=" << lhs.c_str();
          auto array = lhsValuesWithIndex.back()
                           .second.getType()
                           .getElementType()
                           .dyn_cast<hw::ArrayType>();
          hwPort.type =
              builder->getIntegerType(array ? array.getNumElements() : 1);
        } else {
          hwPort.dir = hw::PortInfo::Input;
          rhsValuesWithIndex.push_back({portIdx, convertSigSpec(rhs)});

          if (!rhsValuesWithIndex.back().second)
            return mlir::emitError(location)
                   << "port lowering failed cell name=" << v
                   << " port name=" << lhs.c_str();

          hwPort.type = rhsValuesWithIndex.back().second.getType();
        }

        ports.push_back(hwPort);
      }

      if (v.getValue().starts_with("$")) {
        // Yosys std cells. Just lower it to external module instances.
        auto &extMod = exeternalModules[v];
        if (!extMod) {
          OpBuilder::InsertionGuard guard(*builder);
          builder->setInsertionPointToStart(block.get());
          SmallString<16> name;
          name += "yosys_builtin_cell";
          name += v;
          extMod =
              builder->create<hw::HWModuleExternOp>(location, v, ports, name);
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
    for (auto [str, ext] : mod.exeternalModules) {
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
  ImportRTLILDesign importer{design};
  return importer.run(module);
}
