//===- LowerSeqToSV.cpp - Seq to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq ops to SV.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/Namespace.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace seq;

namespace {
struct SeqToSVPass : public LowerSeqToSVBase<SeqToSVPass> {
  void runOnOperation() override;
};
struct SeqFIRRTLToSVPass : public LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass> {
  void runOnOperation() override;
};
} // anonymous namespace

namespace {
/// Lower CompRegOp to `sv.reg` and `sv.alwaysff`. Use a posedge clock and
/// synchronous reset.
struct CompRegLower : public OpConversionPattern<CompRegOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CompRegOp reg, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = reg.getLoc();

    auto svReg = rewriter.create<sv::RegOp>(loc, reg.getResult().getType(),
                                            reg.getNameAttr());
    svReg->setDialectAttrs(reg->getDialectAttrs());

    // If the seq::CompRegOp has an inner_sym attribute, set this for the
    // sv::RegOp inner_sym attribute.
    if (reg.getSymName().has_value())
      svReg.setInnerSymAttr(reg.getSymNameAttr());

    if (auto attribute = circt::sv::getSVAttributes(reg))
      circt::sv::setSVAttributes(svReg, attribute);

    auto regVal = rewriter.create<sv::ReadInOutOp>(loc, svReg);
    if (reg.getReset() && reg.getResetValue()) {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.getClk(), ResetType::SyncReset,
          sv::EventControl::AtPosEdge, reg.getReset(),
          [&]() { rewriter.create<sv::PAssignOp>(loc, svReg, reg.getInput()); },
          [&]() {
            rewriter.create<sv::PAssignOp>(loc, svReg, reg.getResetValue());
          });
    } else {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.getClk(), [&]() {
            rewriter.create<sv::PAssignOp>(loc, svReg, reg.getInput());
          });
    }

    rewriter.replaceOp(reg, {regVal});
    return success();
  }
};
} // namespace

constexpr uint64_t randomWidth = 32;

namespace {
/// Lower FirRegOp to `sv.reg` and `sv.always`.
class FirRegLower {
public:
  FirRegLower(hw::HWModuleOp module) : module(module), ns(module) {}

  void lower();

  using SymbolAndRange = std::pair<Attribute, std::pair<uint64_t, uint64_t>>;

private:
  using AsyncResetSignal = std::pair<Value, Value>;

  std::pair<sv::RegOp, llvm::Optional<AsyncResetSignal>> lower(FirRegOp reg);

  bool isInitialisePreset(sv::RegOp);

  void initialisePreset(OpBuilder &regBuilder, OpBuilder &initBuilder,
                        sv::RegOp reg);

  void initialise(OpBuilder &regBuilder, OpBuilder &initBuilder, sv::RegOp reg);

  void addToAlwaysBlock(sv::EventControl clockEdge, Value clock,
                        std::function<void(OpBuilder &)> body,
                        ResetType resetStyle = {},
                        sv::EventControl resetEdge = {}, Value reset = {},
                        std::function<void(OpBuilder &)> resetBody = {});

  using AlwaysKeyType = std::tuple<Block *, sv::EventControl, Value, ResetType,
                                   sv::EventControl, Value>;
  llvm::SmallDenseMap<AlwaysKeyType, std::pair<sv::AlwaysOp, sv::IfOp>>
      alwaysBlocks;

  hw::HWModuleOp module;
  hw::ModuleNamespace ns;

  /// This is a map from block to a pair of a random value and its unused
  /// bits. It is used to reduce the number of random value.
  std::pair<Value, uint64_t> randomValueAndRemain;
  SmallDenseMap<StringAttr, sv::RegOp> presetRandomValues;
};
} // namespace

void FirRegLower::lower() {
  // Find all registers to lower in the module.
  auto regs = module.getOps<seq::FirRegOp>();
  if (regs.empty())
    return;

  // Lower the regs to SV regs.
  SmallVector<std::pair<sv::RegOp, llvm::Optional<AsyncResetSignal>>> toInit;
  for (auto reg : llvm::make_early_inc_range(regs))
    toInit.push_back(lower(reg));

  // Create an initial block at the end of the module where random
  // initialisation will be inserted.  Create two builders into the two
  // `ifdef` ops where the registers will be placed.
  //
  // `ifdef SYNTHESIS
  //   `ifdef RANDOMIZE_REG_INIT
  //      ... regBuilder ...
  //   `endif
  //   initial
  //     `INIT_RANDOM_PROLOG_
  //     ... initBuilder ..
  // `endif
  if (!toInit.empty()) {
    auto loc = module.getLoc();
    MLIRContext *context = module.getContext();
    auto randInitRef = sv::MacroIdentAttr::get(context, "RANDOMIZE_REG_INIT");

    auto builder =
        ImplicitLocOpBuilder::atBlockTerminator(loc, module.getBodyBlock());
    builder.create<sv::IfDefOp>(
        "SYNTHESIS", [] {},
        [&] {
          builder.create<sv::OrderedOutputOp>([&] {
            auto ifInitOp = builder.create<sv::IfDefOp>(randInitRef);
            auto regBuilder =
                ImplicitLocOpBuilder::atBlockEnd(loc, ifInitOp.getThenBlock());

            builder.create<sv::IfDefOp>("FIRRTL_BEFORE_INITIAL", [&] {
              builder.create<sv::VerbatimOp>("`FIRRTL_BEFORE_INITIAL");
            });

            builder.create<sv::InitialOp>([&] {
              builder.create<sv::IfDefProceduralOp>("INIT_RANDOM_PROLOG_", [&] {
                builder.create<sv::VerbatimOp>("`INIT_RANDOM_PROLOG_");
              });
              llvm::MapVector<Value, SmallVector<std::pair<sv::RegOp, Value>>>
                  resets;
              builder.create<sv::IfDefProceduralOp>(randInitRef, [&] {
                // Create initialisers for all registers.
                for (auto &[svReg, asyncReset] : toInit) {
                  if (isInitialisePreset(svReg))
                    initialisePreset(regBuilder, builder, svReg);
                  else
                    initialise(regBuilder, builder, svReg);

                  if (asyncReset) {
                    auto &[resetSignal, resetValue] = *asyncReset;
                    resets[resetSignal].emplace_back(svReg, resetValue);
                  }
                }
              });

              if (!resets.empty()) {
                builder.create<sv::IfDefProceduralOp>("RANDOMIZE", [&] {
                  // If the register is async reset, we need to insert extra
                  // initialization in post-randomization so that we can set the
                  // reset value to register if the reset signal is enabled.
                  for (auto &reset : resets) {
                    // Create a block guarded by the RANDOMIZE macro and the
                    // reset: `ifdef RANDOMIZE
                    //   if (reset) begin
                    //     ..
                    //   end
                    // `endif
                    builder.create<sv::IfOp>(reset.first, [&] {
                      for (auto &[reg, value] : reset.second)
                        builder.create<sv::BPAssignOp>(reg.getLoc(), reg,
                                                       value);
                    });
                  }
                });
              }
            });

            builder.create<sv::IfDefOp>("FIRRTL_AFTER_INITIAL", [&] {
              builder.create<sv::VerbatimOp>("`FIRRTL_AFTER_INITIAL");
            });
          });
        });
  }

  module->removeAttr("firrtl.random_init_width");
}

std::pair<sv::RegOp, llvm::Optional<FirRegLower::AsyncResetSignal>>
FirRegLower::lower(FirRegOp reg) {
  Location loc = reg.getLoc();

  ImplicitLocOpBuilder builder(reg.getLoc(), reg);
  auto svReg = builder.create<sv::RegOp>(loc, reg.getResult().getType(),
                                         reg.getNameAttr());
  svReg->setDialectAttrs(reg->getDialectAttrs());

  if (auto innerSymAttr = reg.getInnerSymAttr())
    svReg.setInnerSymAttr(innerSymAttr);
  else
    svReg.setInnerSymAttr(
        builder.getStringAttr(ns.newName(Twine("__") + reg.getName() + "__")));

  auto regVal = builder.create<sv::ReadInOutOp>(loc, svReg);

  auto setInput = [&](OpBuilder &builder) {
    if (reg.getNext() != reg)
      builder.create<sv::PAssignOp>(loc, svReg, reg.getNext());
  };

  llvm::Optional<AsyncResetSignal> asyncReset;
  if (reg.hasReset()) {
    addToAlwaysBlock(
        sv::EventControl::AtPosEdge, reg.getClk(), setInput,
        reg.getIsAsync() ? ResetType::AsyncReset : ResetType::SyncReset,
        sv::EventControl::AtPosEdge, reg.getReset(), [&](OpBuilder &builder) {
          builder.create<sv::PAssignOp>(loc, svReg, reg.getResetValue());
        });

    if (reg.getIsAsync()) {
      asyncReset = std::make_pair(reg.getReset(), reg.getResetValue());
    }
  } else {
    addToAlwaysBlock(sv::EventControl::AtPosEdge, reg.getClk(), setInput);
  }

  reg.replaceAllUsesWith(regVal.getResult());
  reg.erase();

  return {svReg, asyncReset};
}

static void emitRandomInit(
    hw::HWModuleOp module, sv::RegOp reg, OpBuilder &builder,
    uint64_t randomRegWidth,
    llvm::function_ref<void(IntegerType,
                            SmallVector<FirRegLower::SymbolAndRange> &)>
        getRandomValues) {
  auto regDefSym =
      hw::InnerRefAttr::get(module.getNameAttr(), reg.getInnerSymAttr());

  // Get a random value with the specified width, combining or truncating
  // 32-bit units as necessary.
  auto emitRandomInit = [&](Value dest, Type type, const Twine &accessor) {
    auto intType = type.cast<IntegerType>();
    if (intType.getWidth() == 0)
      return;

    SmallVector<FirRegLower::SymbolAndRange> values;
    getRandomValues(intType, values);

    SmallString<32> rhs(("{{0}}" + accessor + " = ").str());
    unsigned i = 1;
    SmallVector<Attribute, 4> symbols({regDefSym});
    if (values.size() > 1)
      rhs.append("{");
    for (auto [value, range] : llvm::reverse(values)) {
      symbols.push_back(value);
      auto [high, low] = range;
      if (i > 1)
        rhs.append(", ");
      rhs.append(("{{" + Twine(i++) + "}}").str());

      // This uses all bits of the random value. Emit without part select.
      if (high == randomWidth - 1 && low == 0 && randomRegWidth == randomWidth)
        continue;

      // Emit a single bit part select, e.g., emit "[0]" and not "[0:0]".
      if (high == low) {
        rhs.append(("[" + Twine(high) + "]").str());
        continue;
      }

      // Emit a part select, e.g., "[4:2]"
      rhs.append(
          ("[" + Twine(range.first) + ":" + Twine(range.second) + "]").str());
    }
    if (values.size() > 1)
      rhs.append("}");
    rhs.append(";");

    builder.create<sv::VerbatimOp>(reg.getLoc(), rhs, ValueRange{},
                                   builder.getArrayAttr(symbols));
  };

  // Randomly initialize everything in the register. If the register
  // is an aggregate type, then assign random values to all its
  // constituent ground types.
  auto type = reg.getType().dyn_cast<hw::InOutType>().getElementType();
  std::function<void(Type, const Twine &)> recurse = [&](Type type,
                                                         const Twine &member) {
    TypeSwitch<Type>(type)
        .Case<hw::UnpackedArrayType, hw::ArrayType>([&](auto a) {
          for (size_t i = 0, e = a.getSize(); i != e; ++i)
            recurse(a.getElementType(), member + "[" + Twine(i) + "]");
        })
        .Case<hw::StructType>([&](hw::StructType s) {
          for (auto elem : s.getElements())
            recurse(elem.type, member + "." + elem.name.getValue());
        })
        .Default([&](auto type) { emitRandomInit(reg, type, member); });
  };
  recurse(type, "");
}

bool FirRegLower::isInitialisePreset(sv::RegOp reg) {
  return module->hasAttr("firrtl.random_init_width") &&
         reg->hasAttr("firrtl.random_init_register") &&
         reg->hasAttr("firrtl.random_init_start") &&
         reg->hasAttr("firrtl.random_init_end");
}

void FirRegLower::initialisePreset(OpBuilder &regBuilder,
                                   OpBuilder &initBuilder, sv::RegOp reg) {
  // Extract the required random initialisation width and compute the number of
  // `RANDOM calls required to fill it up, as well as the final register width.
  auto randomInitWidthDict =
      module->getAttrOfType<DictionaryAttr>("firrtl.random_init_width");
  assert(randomInitWidthDict &&
         "firrtl.random_init_width required for preset initialisation");

  auto randomRegister =
      reg->getAttrOfType<StringAttr>("firrtl.random_init_register");
  assert(randomRegister &&
         "firrtl.random_init_register required for preset initialisation");

  // Only create the random register(s) once.
  if (presetRandomValues.empty()) {
    for (NamedAttribute registerAndWidth : randomInitWidthDict.getValue()) {
      auto randomInitWidth =
          registerAndWidth.getValue().cast<IntegerAttr>().getUInt();
      assert(randomInitWidth > 0 &&
             "random initialization width should be non-zero");
      uint64_t numRandomSources =
          llvm::divideCeil(randomInitWidth, randomWidth);
      uint64_t randomRegWidth = randomWidth * numRandomSources;

      // Declare the random register.
      auto randReg = regBuilder.create<sv::RegOp>(
          reg.getLoc(), regBuilder.getIntegerType(randomRegWidth),
          /*name=*/regBuilder.getStringAttr("_RANDOM"),
          /*inner_sym=*/
          regBuilder.getStringAttr(ns.newName(Twine("_RANDOM"))));

      presetRandomValues[registerAndWidth.getName()] = randReg;

      SmallString<32> randomRegAssign("{{0}} = {");

      // Fill the random register by concatenating calls to `RANDOM in a
      // verbatim string.
      for (uint64_t i = 0; i < numRandomSources; ++i) {
        randomRegAssign.append("`RANDOM");
        if (i < numRandomSources - 1)
          randomRegAssign.append(",");
        // Add a line break when line length gets close to 1000 characters.
        if (i > 0 && i % 125 == 0)
          randomRegAssign.append("\n");
      }

      randomRegAssign.append("};");

      // Assign the concatenated calls to the declared register.
      initBuilder.create<sv::VerbatimOp>(
          reg.getLoc(), initBuilder.getStringAttr(randomRegAssign),
          ValueRange{},
          initBuilder.getArrayAttr({hw::InnerRefAttr::get(
              module.getNameAttr(), randReg.getInnerSymAttr())}));
    }
  }

  auto getRandomValues = [&](IntegerType type,
                             SmallVector<SymbolAndRange> &values) {
    assert(type.getWidth() != 0 && "zero bit width's not supported");

    auto randomStart =
        reg->getAttrOfType<IntegerAttr>("firrtl.random_init_start").getUInt();
    auto randomEnd =
        reg->getAttrOfType<IntegerAttr>("firrtl.random_init_end").getUInt();
    reg->removeAttr("firrtl.random_init_start");
    reg->removeAttr("firrtl.random_init_end");

    auto randReg = presetRandomValues[randomRegister];

    auto symbol =
        hw::InnerRefAttr::get(module.getNameAttr(), randReg.getInnerSymAttr());

    values.push_back({symbol, {randomEnd, randomStart}});
  };

  auto randomRegWidth = presetRandomValues[randomRegister]
                            .getElementType()
                            .getIntOrFloatBitWidth();
  emitRandomInit(module, reg, initBuilder, randomRegWidth, getRandomValues);
}

void FirRegLower::initialise(OpBuilder &regBuilder, OpBuilder &initBuilder,
                             sv::RegOp reg) {
  // Construct and return a new reference to `RANDOM.  It is always a 32-bit
  // unsigned expression.  Calls to $random have side effects, so we use
  // VerbatimExprSEOp.
  auto getRandom32Val = [&](const char *suffix = "") -> Value {
    sv::RegOp randReg = regBuilder.create<sv::RegOp>(
        reg.getLoc(), regBuilder.getIntegerType(randomWidth),
        /*name=*/regBuilder.getStringAttr("_RANDOM"),
        /*inner_sym=*/
        regBuilder.getStringAttr(ns.newName("_RANDOM")));

    initBuilder.create<sv::VerbatimOp>(
        reg.getLoc(), initBuilder.getStringAttr("{{0}} = {`RANDOM};"),
        ValueRange{},
        initBuilder.getArrayAttr({hw::InnerRefAttr::get(
            module.getNameAttr(), randReg.getInnerSymAttr())}));

    return randReg.getResult();
  };

  auto getRandomValues = [&](IntegerType type,
                             SmallVector<SymbolAndRange> &values) {
    auto width = type.getWidth();
    assert(width != 0 && "zero bit width's not supported");
    while (width > 0) {
      // If there are no bits left, then generate a new random value.
      if (!randomValueAndRemain.second)
        randomValueAndRemain = {getRandom32Val("foo"), randomWidth};

      auto reg = cast<sv::RegOp>(randomValueAndRemain.first.getDefiningOp());

      auto symbol =
          hw::InnerRefAttr::get(module.getNameAttr(), reg.getInnerSymAttr());
      unsigned low = randomWidth - randomValueAndRemain.second;
      unsigned high = randomWidth - 1;
      if (width <= randomValueAndRemain.second)
        high = width - 1 + low;
      unsigned consumed = high - low + 1;
      values.push_back({symbol, {high, low}});
      randomValueAndRemain.second -= consumed;
      width -= consumed;
    }
  };

  emitRandomInit(module, reg, initBuilder, randomWidth, getRandomValues);
}

void FirRegLower::addToAlwaysBlock(sv::EventControl clockEdge, Value clock,
                                   std::function<void(OpBuilder &)> body,
                                   ::ResetType resetStyle,
                                   sv::EventControl resetEdge, Value reset,
                                   std::function<void(OpBuilder &)> resetBody) {
  auto loc = module.getLoc();
  auto builder =
      ImplicitLocOpBuilder::atBlockTerminator(loc, module.getBodyBlock());

  auto &op = alwaysBlocks[{builder.getBlock(), clockEdge, clock, resetStyle,
                           resetEdge, reset}];
  auto &alwaysOp = op.first;
  auto &insideIfOp = op.second;

  if (!alwaysOp) {
    if (reset) {
      assert(resetStyle != ::ResetType::NoReset);
      // Here, we want to create the following structure with sv.always and
      // sv.if. If `reset` is async, we need to add `reset` to a sensitivity
      // list.
      //
      // sv.always @(clockEdge or reset) {
      //   sv.if (reset) {
      //     resetBody
      //   } else {
      //     body
      //   }
      // }

      auto createIfOp = [&]() {
        // It is weird but intended. Here we want to create an empty sv.if
        // with an else block.
        insideIfOp = builder.create<sv::IfOp>(
            reset, []() {}, []() {});
      };
      if (resetStyle == ::ResetType::AsyncReset) {
        sv::EventControl events[] = {clockEdge, resetEdge};
        Value clocks[] = {clock, reset};

        alwaysOp = builder.create<sv::AlwaysOp>(events, clocks, [&]() {
          if (resetEdge == sv::EventControl::AtNegEdge)
            llvm_unreachable("negative edge for reset is not expected");
          createIfOp();
        });
      } else {
        alwaysOp = builder.create<sv::AlwaysOp>(clockEdge, clock, createIfOp);
      }
    } else {
      assert(!resetBody);
      alwaysOp = builder.create<sv::AlwaysOp>(clockEdge, clock);
      insideIfOp = nullptr;
    }
  }

  if (reset) {
    assert(insideIfOp && "reset body must be initialized before");
    auto resetBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, insideIfOp.getThenBlock());
    resetBody(resetBuilder);

    auto bodyBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, insideIfOp.getElseBlock());
    body(bodyBuilder);
  } else {
    auto bodyBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, alwaysOp.getBodyBlock());
    body(bodyBuilder);
  }
}

void SeqToSVPass::runOnOperation() {
  ModuleOp top = getOperation();

  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);
  target.addIllegalDialect<SeqDialect>();
  target.addLegalDialect<sv::SVDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<CompRegLower>(&ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

void SeqFIRRTLToSVPass::runOnOperation() {
  ModuleOp top = getOperation();

  for (auto module : top.getOps<hw::HWModuleOp>())
    FirRegLower(module).lower();
}

std::unique_ptr<Pass> circt::seq::createSeqLowerToSVPass() {
  return std::make_unique<SeqToSVPass>();
}

std::unique_ptr<Pass> circt::seq::createSeqFIRRTLLowerToSVPass() {
  return std::make_unique<SeqFIRRTLToSVPass>();
}
