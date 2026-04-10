//===- LowerSimToSV.cpp - Sim to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translates Sim ops to SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SimToSV.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "lower-sim-to-sv"

namespace circt {
#define GEN_PASS_DEF_LOWERSIMTOSV
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace sim;

/// Check whether an op should be placed inside an ifdef guard that prevents it
/// from affecting synthesis runs.
static bool needsIfdefGuard(Operation *op) {
  return isa<ClockedTerminateOp, ClockedPauseOp, TerminateOp, PauseOp,
             PrintFormattedProcOp, FFlushProcOp>(op);
}

/// Check whether an op should be placed inside an always process triggered on a
/// clock, and an if statement checking for a condition.
static std::pair<Value, Value> needsClockAndConditionWrapper(Operation *op) {
  return TypeSwitch<Operation *, std::pair<Value, Value>>(op)
      .Case<ClockedTerminateOp, ClockedPauseOp>(
          [](auto op) -> std::pair<Value, Value> {
            return {op.getClock(), op.getCondition()};
          })
      .Default({});
}

static Value getDefaultStderrFD(OpBuilder &builder, Location loc) {
  return hw::ConstantOp::create(builder, loc, APInt(32, 0x80000002));
}

struct LoweredFormatString {
  SmallString<64> format;
  SmallVector<Value> operands;
};

static void appendEscapedLiteral(SmallString<64> &result, StringRef literal) {
  for (char c : literal) {
    if (c == '%')
      result += "%%";
    else
      result.push_back(c);
  }
}

static void appendDecimalSpecifier(SmallString<64> &result,
                                   std::optional<uint32_t> width) {
  result.push_back('%');
  if (width)
    result += Twine(*width).str();
  result.push_back('d');
}

static LogicalResult lowerFormatString(Value input, LoweredFormatString &out,
                                       ConversionPatternRewriter &rewriter) {
  Operation *op = input.getDefiningOp();
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<FormatLiteralOp>([&](auto lit) {
        appendEscapedLiteral(out.format, lit.getLiteral());
        return success();
      })
      .Case<FormatStringConcatOp>([&](auto concat) {
        for (Value sub : concat.getInputs())
          if (failed(lowerFormatString(sub, out, rewriter)))
            return failure();
        return success();
      })
      .Case<FormatBinOp>([&](auto fmt) {
        out.format.push_back('%');
        if (fmt.getSpecifierWidth())
          out.format += Twine(*fmt.getSpecifierWidth()).str();
        out.format.push_back('b');
        out.operands.push_back(fmt.getValue());
        return success();
      })
      .Case<FormatDecOp>([&](auto fmt) {
        if (fmt.getValue().template getDefiningOp<sim::TimeOp>()) {
          out.format.push_back('%');
          if (fmt.getSpecifierWidth())
            out.format += Twine(*fmt.getSpecifierWidth()).str();
          else
            out.format.push_back('0');
          out.format.push_back('t');
          out.operands.push_back(fmt.getValue());
        } else {
          appendDecimalSpecifier(out.format, fmt.getSpecifierWidth());

          Value operand = fmt.getValue();
          if (fmt.getIsSigned()) {
            operand = sv::SystemFunctionOp::create(
                rewriter, fmt.getLoc(), operand.getType(), "signed", operand);
          }
          out.operands.push_back(operand);
        }
        return success();
      })
      .Case<FormatHexOp>([&](auto fmt) {
        out.format.push_back('%');
        if (fmt.getSpecifierWidth())
          out.format += Twine(*fmt.getSpecifierWidth()).str();
        out.format.push_back(fmt.getIsHexUppercase() ? 'X' : 'x');
        out.operands.push_back(fmt.getValue());
        return success();
      })
      .Case<FormatOctOp>([&](auto fmt) {
        out.format.push_back('%');
        if (fmt.getSpecifierWidth())
          out.format += Twine(*fmt.getSpecifierWidth()).str();
        out.format.push_back('o');
        out.operands.push_back(fmt.getValue());
        return success();
      })
      .Case<FormatCharOp>([&](auto fmt) {
        out.format += "%c";
        out.operands.push_back(fmt.getValue());
        return success();
      })
      .Case<FormatHierPathOp>([&](auto fmt) {
        out.format += fmt.getUseEscapes() ? "%M" : "%m";
        return success();
      })
      .Default([](Operation *) { return failure(); });
}

static bool isFormatOp(Operation *op) {
  return isa<FormatLiteralOp, FormatStringConcatOp, FormatBinOp, FormatDecOp,
             FormatHexOp, FormatOctOp, FormatCharOp, FormatHierPathOp,
             FormatScientificOp, FormatFloatOp, FormatGeneralOp>(op);
}

namespace {

struct SimConversionState {
  hw::HWModuleOp module;
  bool usedSynthesisMacro = false;
  SetVector<StringAttr> dpiCallees;
  SmallVector<Value> formatCleanupRoots;
};

template <typename T>
struct SimConversionPattern : public OpConversionPattern<T> {
  explicit SimConversionPattern(MLIRContext *context, SimConversionState &state)
      : OpConversionPattern<T>(context), state(state) {}

  SimConversionState &state;
};

hw::ModuleType
DPIFunctionTypeToHWModuleType(const DPIFunctionType &dpiFuncType) {
  SmallVector<hw::ModulePort> hwPorts;
  for (auto &arg : dpiFuncType.getArguments()) {
    hw::ModulePort::Direction hwDir;
    switch (arg.dir) {
    case DPIDirection::Input:
    case DPIDirection::Ref:
      hwDir = hw::ModulePort::Direction::Input;
      break;
    case DPIDirection::Output:
    case DPIDirection::Return:
      hwDir = hw::ModulePort::Direction::Output;
      break;
    case DPIDirection::InOut:
      hwDir = hw::ModulePort::Direction::InOut;
      break;
    }
    hwPorts.push_back({arg.name, arg.type, hwDir});
  }
  return hw::ModuleType::get(dpiFuncType.getContext(), hwPorts);
}
} // namespace

// Lower `sim.plusargs.test` to a standard SV implementation.
//
class PlusArgsTestLowering : public SimConversionPattern<PlusArgsTestOp> {
public:
  using SimConversionPattern<PlusArgsTestOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(PlusArgsTestOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto resultType = rewriter.getIntegerType(1);
    auto str = sv::ConstantStrOp::create(rewriter, loc, op.getFormatString());
    auto reg = sv::RegOp::create(rewriter, loc, resultType,
                                 rewriter.getStringAttr("_pargs"));
    sv::InitialOp::create(rewriter, loc, [&] {
      auto call = sv::SystemFunctionOp::create(
          rewriter, loc, resultType, "test$plusargs", ArrayRef<Value>{str});
      sv::BPAssignOp::create(rewriter, loc, reg, call);
    });

    rewriter.replaceOpWithNewOp<sv::ReadInOutOp>(op, reg);
    return success();
  }
};

// Lower `sim.plusargs.value` to a standard SV implementation.
//
class PlusArgsValueLowering : public SimConversionPattern<PlusArgsValueOp> {
public:
  using SimConversionPattern<PlusArgsValueOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(PlusArgsValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    auto i1ty = rewriter.getIntegerType(1);
    auto type = op.getResult().getType();

    auto wirev = sv::WireOp::create(rewriter, loc, type,
                                    rewriter.getStringAttr("_pargs_v"));
    auto wiref = sv::WireOp::create(rewriter, loc, i1ty,
                                    rewriter.getStringAttr("_pargs_f"));

    state.usedSynthesisMacro = true;
    sv::IfDefOp::create(
        rewriter, loc, "SYNTHESIS",
        [&]() {
          auto cstFalse = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));
          auto cstZ = sv::ConstantZOp::create(rewriter, loc, type);
          auto assignZ = sv::AssignOp::create(rewriter, loc, wirev, cstZ);
          circt::sv::setSVAttributes(
              assignZ,
              sv::SVAttributeAttr::get(
                  rewriter.getContext(),
                  "This dummy assignment exists to avoid undriven lint "
                  "warnings (e.g., Verilator UNDRIVEN).",
                  /*emitAsComment=*/true));
          sv::AssignOp::create(rewriter, loc, wiref, cstFalse);
        },
        [&]() {
          auto i32ty = rewriter.getIntegerType(32);
          auto regf = sv::RegOp::create(rewriter, loc, i32ty,
                                        rewriter.getStringAttr("_found"));
          auto regv = sv::RegOp::create(rewriter, loc, type,
                                        rewriter.getStringAttr("_value"));
          sv::InitialOp::create(rewriter, loc, [&] {
            auto str =
                sv::ConstantStrOp::create(rewriter, loc, op.getFormatString());
            auto call = sv::SystemFunctionOp::create(
                rewriter, loc, i32ty, "value$plusargs",
                ArrayRef<Value>{str, regv});
            sv::BPAssignOp::create(rewriter, loc, regf, call);
          });
          Value readRegF = sv::ReadInOutOp::create(rewriter, loc, regf);
          Value readRegV = sv::ReadInOutOp::create(rewriter, loc, regv);
          auto cstTrue = hw::ConstantOp::create(rewriter, loc, i32ty, 1);
          // Squash any X coming from the regf to 0.
          auto cmp = comb::ICmpOp::create(
              rewriter, loc, comb::ICmpPredicate::ceq, readRegF, cstTrue);
          sv::AssignOp::create(rewriter, loc, wiref, cmp);
          sv::AssignOp::create(rewriter, loc, wirev, readRegV);
        });

    Value readf = sv::ReadInOutOp::create(rewriter, loc, wiref);
    Value readv = sv::ReadInOutOp::create(rewriter, loc, wirev);

    rewriter.replaceOp(op, {readf, readv});
    return success();
  }
};

static LogicalResult convert(ClockedTerminateOp op, PatternRewriter &rewriter) {
  if (op.getSuccess())
    rewriter.replaceOpWithNewOp<sv::FinishOp>(op, op.getVerbose());
  else
    rewriter.replaceOpWithNewOp<sv::FatalProceduralOp>(op, op.getVerbose());
  return success();
}

static LogicalResult convert(ClockedPauseOp op, PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<sv::StopOp>(op, op.getVerbose());
  return success();
}

static LogicalResult convert(TerminateOp op, PatternRewriter &rewriter) {
  if (op.getSuccess())
    rewriter.replaceOpWithNewOp<sv::FinishOp>(op, op.getVerbose());
  else
    rewriter.replaceOpWithNewOp<sv::FatalProceduralOp>(op, op.getVerbose());
  return success();
}

static LogicalResult convert(PauseOp op, PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<sv::StopOp>(op, op.getVerbose());
  return success();
}

class DPICallLowering : public SimConversionPattern<DPICallOp> {
public:
  using SimConversionPattern<DPICallOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(DPICallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    // Record the callee.
    state.dpiCallees.insert(op.getCalleeAttr().getAttr());

    bool isClockedCall = !!op.getClock();
    bool hasEnable = !!op.getEnable();

    SmallVector<sv::RegOp> temporaries;
    SmallVector<Value> reads;
    for (auto [type, result] :
         llvm::zip(op.getResultTypes(), op.getResults())) {
      temporaries.push_back(sv::RegOp::create(rewriter, op.getLoc(), type));
      reads.push_back(
          sv::ReadInOutOp::create(rewriter, op.getLoc(), temporaries.back()));
    }

    auto emitCall = [&]() {
      auto call = sv::FuncCallProceduralOp::create(
          rewriter, op.getLoc(), op.getResultTypes(), op.getCalleeAttr(),
          adaptor.getInputs());
      for (auto [lhs, rhs] : llvm::zip(temporaries, call.getResults())) {
        if (isClockedCall)
          sv::PAssignOp::create(rewriter, op.getLoc(), lhs, rhs);
        else
          sv::BPAssignOp::create(rewriter, op.getLoc(), lhs, rhs);
      }
    };
    if (isClockedCall) {
      Value clockCast =
          seq::FromClockOp::create(rewriter, loc, adaptor.getClock());
      sv::AlwaysOp::create(
          rewriter, loc,
          ArrayRef<sv::EventControl>{sv::EventControl::AtPosEdge},
          ArrayRef<Value>{clockCast}, [&]() {
            if (!hasEnable)
              return emitCall();
            sv::IfOp::create(rewriter, op.getLoc(), adaptor.getEnable(),
                             emitCall);
          });
    } else {
      // Unclocked call is lowered into always_comb.
      // TODO: If there is a return value and no output argument, use an
      // unclocked call op.
      sv::AlwaysCombOp::create(rewriter, loc, [&]() {
        if (!hasEnable)
          return emitCall();
        auto assignXToResults = [&] {
          for (auto lhs : temporaries) {
            auto xValue = sv::ConstantXOp::create(
                rewriter, op.getLoc(), lhs.getType().getElementType());
            sv::BPAssignOp::create(rewriter, op.getLoc(), lhs, xValue);
          }
        };
        sv::IfOp::create(rewriter, op.getLoc(), adaptor.getEnable(), emitCall,
                         assignXToResults);
      });
    }

    rewriter.replaceOp(op, reads);
    return success();
  }
};

class GetFileLowering : public SimConversionPattern<GetFileOp> {
public:
  using SimConversionPattern<GetFileOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(GetFileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value fileName;
    if (adaptor.getFileNameOperands().empty()) {
      fileName = sv::ConstantStrOp::create(rewriter, op.getLoc(),
                                           op.getFileNameFormat());
    } else {
      fileName =
          sv::SFormatFOp::create(rewriter, op.getLoc(), op.getFileNameFormat(),
                                 adaptor.getFileNameOperands());
    }

    Value fd;
    auto callee =
        rewriter.getStringAttr("__circt_lib_logging::FileDescriptor::get");
    if (op->getParentOp() &&
        op->getParentOp()->hasTrait<sv::ProceduralRegion>()) {
      fd = sv::FuncCallProceduralOp::create(rewriter, op.getLoc(),
                                            TypeRange{rewriter.getI32Type()},
                                            callee, ValueRange{fileName})
               ->getResult(0);
    } else {
      fd = sv::FuncCallOp::create(rewriter, op.getLoc(),
                                  TypeRange{rewriter.getI32Type()}, callee,
                                  ValueRange{fileName})
               ->getResult(0);
    }
    rewriter.replaceOp(op, fd);
    return success();
  }
};

class PrintFormattedProcLowering
    : public SimConversionPattern<PrintFormattedProcOp> {
public:
  using SimConversionPattern<PrintFormattedProcOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(PrintFormattedProcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value originalInput = op.getInput();

    LoweredFormatString lowered;
    if (failed(lowerFormatString(adaptor.getInput(), lowered, rewriter)))
      return rewriter.notifyMatchFailure(op, "unsupported format string");

    Value stream = adaptor.getStream();
    if (!stream)
      stream = getDefaultStderrFD(rewriter, op.getLoc());

    auto buildFwrite = [&]() {
      sv::FWriteOp::create(rewriter, op.getLoc(), stream,
                           rewriter.getStringAttr(lowered.format),
                           lowered.operands);
    };

    if (op.getUsePrintfCond()) {
      Value printfCond = sv::MacroRefExprOp::create(
          rewriter, op->getLoc(), rewriter.getI1Type(), "PRINTF_COND_");
      sv::IfOp::create(rewriter, op->getLoc(), printfCond,
                       [&]() { buildFwrite(); });
    } else {
      buildFwrite();
    }
    rewriter.eraseOp(op);

    state.formatCleanupRoots.push_back(originalInput);

    return success();
  }
};

class FFlushProcLowering : public SimConversionPattern<FFlushProcOp> {
public:
  using SimConversionPattern<FFlushProcOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(FFlushProcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    sv::FFlushOp::create(rewriter, op.getLoc(), adaptor.getStream());
    rewriter.eraseOp(op);
    return success();
  }
};

class ScfIfLowering : public OpConversionPattern<mlir::scf::IfOp> {
public:
  using OpConversionPattern<mlir::scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getNumResults() != 0)
      return rewriter.notifyMatchFailure(op, "resultful scf.if unsupported");

    auto *parentOp = op->getParentOp();
    if (!parentOp || !parentOp->hasTrait<sv::ProceduralRegion>())
      return rewriter.notifyMatchFailure(
          op, "scf.if parent is not an SV procedural region");

    auto cloneWithoutYield = [&](Block &from) {
      for (auto &nested : from.without_terminator())
        rewriter.clone(nested);
    };

    if (op.getElseRegion().empty()) {
      sv::IfOp::create(rewriter, op.getLoc(), adaptor.getCondition(), [&]() {
        cloneWithoutYield(op.getThenRegion().front());
      });
    } else {
      sv::IfOp::create(
          rewriter, op.getLoc(), adaptor.getCondition(),
          [&]() { cloneWithoutYield(op.getThenRegion().front()); },
          [&]() { cloneWithoutYield(op.getElseRegion().front()); });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class TimeLowering : public SimConversionPattern<sim::TimeOp> {
public:
  using SimConversionPattern<sim::TimeOp>::SimConversionPattern;

  LogicalResult
  matchAndRewrite(sim::TimeOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<sv::TimeOp>(op);
    return success();
  }
};

// A helper struct to lower DPI function/call.
struct LowerDPIFunc {
  llvm::DenseMap<StringAttr, StringAttr> symbolToFragment;
  circt::Namespace nameSpace;
  LowerDPIFunc(mlir::ModuleOp module) { nameSpace.add(module); }
  void lower(sim::DPIFuncOp func);
  void addFragments(hw::HWModuleOp module,
                    ArrayRef<StringAttr> dpiCallees) const;
};

static ArrayAttr buildSVPerArgumentAttrs(MLIRContext *context,
                                         sim::DPIFuncOp func) {
  Builder builder(context);
  SmallVector<Attribute> convertedAttrs;
  auto dpiType = func.getDpiFunctionType();
  auto dpiArgs = dpiType.getArguments();
  convertedAttrs.reserve(dpiArgs.size());
  for (auto &arg : dpiArgs) {
    NamedAttrList newAttrs;
    if (arg.dir == sim::DPIDirection::Return)
      newAttrs.append(
          builder.getStringAttr(sv::FuncOp::getExplicitlyReturnedAttrName()),
          builder.getUnitAttr());
    convertedAttrs.push_back(newAttrs.getDictionary(context));
  }
  return ArrayAttr::get(context, convertedAttrs);
}

void LowerDPIFunc::lower(sim::DPIFuncOp func) {
  ImplicitLocOpBuilder builder(func.getLoc(), func);
  ArrayAttr inputLocsAttr, outputLocsAttr;

  // Build ModuleType from DPI arguments for sv::FuncOp.
  auto moduleType = DPIFunctionTypeToHWModuleType(func.getDpiFunctionType());

  if (func.getArgumentLocs()) {
    SmallVector<Attribute> inputLocs, outputLocs;
    auto hwPorts = moduleType.getPorts();
    for (auto [port, loc] : llvm::zip(
             hwPorts, func.getArgumentLocsAttr().getAsRange<LocationAttr>())) {
      (port.dir == hw::ModulePort::Output ? outputLocs : inputLocs)
          .push_back(loc);
    }
    inputLocsAttr = builder.getArrayAttr(inputLocs);
    outputLocsAttr = builder.getArrayAttr(outputLocs);
  }

  auto svFuncDecl = sv::FuncOp::create(
      builder, func.getSymNameAttr(), moduleType,
      buildSVPerArgumentAttrs(builder.getContext(), func), inputLocsAttr,
      outputLocsAttr, func.getVerilogNameAttr());
  // DPI function is a declaration so it must be a private function.
  svFuncDecl.setPrivate();
  auto name = builder.getStringAttr(nameSpace.newName(
      func.getSymNameAttr().getValue(), "dpi_import_fragument"));

  // Add include guards to avoid duplicate declarations. See Issue 7458.
  auto macroDecl = sv::MacroDeclOp::create(
      builder, nameSpace.newName("__CIRCT_DPI_IMPORT",
                                 func.getSymNameAttr().getValue().upper()));
  emit::FragmentOp::create(builder, name, [&]() {
    sv::IfDefOp::create(
        builder, macroDecl.getSymNameAttr(), []() {},
        [&]() {
          sv::FuncDPIImportOp::create(builder, func.getSymNameAttr(),
                                      StringAttr());
          sv::MacroDefOp::create(builder, macroDecl.getSymNameAttr(), "");
        });
  });

  symbolToFragment.insert({func.getSymNameAttr(), name});
  func.erase();
}

void LowerDPIFunc::addFragments(hw::HWModuleOp module,
                                ArrayRef<StringAttr> dpiCallees) const {
  llvm::SetVector<Attribute> fragments;
  // Add existing emit fragments.
  if (auto exstingFragments =
          module->getAttrOfType<ArrayAttr>(emit::getFragmentsAttrName()))
    for (auto fragment : exstingFragments.getAsRange<FlatSymbolRefAttr>())
      fragments.insert(fragment);
  for (auto callee : dpiCallees) {
    auto attr = symbolToFragment.at(callee);
    fragments.insert(FlatSymbolRefAttr::get(attr));
  }
  if (!fragments.empty())
    module->setAttr(
        emit::getFragmentsAttrName(),
        ArrayAttr::get(module.getContext(), fragments.takeVector()));
}

static bool isInProceduralRegion(Operation *op) {
  auto *parentOp = op->getParentOp();
  while (parentOp && !parentOp->hasTrait<sv::ProceduralRegion>()) {
    parentOp = parentOp->getParentOp();
  }
  return parentOp;
}

static bool moveOpsIntoIfdefGuardsAndProcesses(Operation *rootOp) {
  bool usedSynthesisMacro = false;

  rootOp->walk([&](Operation *op) {
    auto loc = op->getLoc();

    // Move the op into an ifdef guard if needed.
    if (needsIfdefGuard(op)) {
      // Try to reuse an ifdef guard immediately before the op.
      bool isProcedural = isInProceduralRegion(op);
      Block *block = nullptr;
      if (op->getPrevNode())
        block = TypeSwitch<Operation *, Block *>(op->getPrevNode())
                    .Case<sv::IfDefOp, sv::IfDefProceduralOp>(
                        [&](auto guardOp) -> Block * {
                          bool guardIsProcedural = isa<sv::IfDefProceduralOp>(
                              guardOp.getOperation());
                          if (guardIsProcedural != isProcedural)
                            return nullptr;
                          if (guardOp.getCond().getIdent().getAttr() ==
                                  "SYNTHESIS" &&
                              guardOp.hasElse())
                            return guardOp.getElseBlock();
                          return nullptr;
                        })
                    .Default([](auto) { return nullptr; });

      // If there was no pre-existing guard, create one.
      if (!block) {
        OpBuilder builder(op);
        if (isProcedural)
          block = sv::IfDefProceduralOp::create(
                      builder, loc, "SYNTHESIS", [] {}, [] {})
                      .getElseBlock();
        else
          block = sv::IfDefOp::create(
                      builder, loc, "SYNTHESIS", [] {}, [] {})
                      .getElseBlock();
        usedSynthesisMacro = true;
      }

      // Move the op into the guard block.
      op->moveBefore(block, block->end());
    }

    // Check if the op requires an clock and condition wrapper.
    auto [clock, condition] = needsClockAndConditionWrapper(op);

    // Create an enclosing always process.
    if (clock) {
      // Try to reuse an always process immediately before the op.
      Block *block = nullptr;
      if (auto alwaysOp = dyn_cast_or_null<sv::AlwaysOp>(op->getPrevNode()))
        if (alwaysOp.getNumConditions() == 1 &&
            alwaysOp.getCondition(0).event == sv::EventControl::AtPosEdge)
          if (auto clockOp = alwaysOp.getCondition(0)
                                 .value.getDefiningOp<seq::FromClockOp>())
            if (clockOp.getInput() == clock)
              block = alwaysOp.getBodyBlock();

      // If there was no pre-existing always process, create one.
      if (!block) {
        OpBuilder builder(op);
        clock = seq::FromClockOp::create(builder, loc, clock);
        block = sv::AlwaysOp::create(builder, loc, sv::EventControl::AtPosEdge,
                                     clock, [] {})
                    .getBodyBlock();
      }

      // Move the op into the process.
      op->moveBefore(block, block->end());
    }

    // Create an enclosing if condition.
    if (condition) {
      // Try to reuse an if statement immediately before the op.
      Block *block = nullptr;
      if (auto ifOp = dyn_cast_or_null<sv::IfOp>(op->getPrevNode()))
        if (ifOp.getCond() == condition)
          block = ifOp.getThenBlock();

      // If there was no pre-existing if statement, create one.
      if (!block) {
        OpBuilder builder(op);
        block = sv::IfOp::create(builder, loc, condition, [] {}).getThenBlock();
      }

      // Move the op into the if body.
      op->moveBefore(block, block->end());
    }
  });

  return usedSynthesisMacro;
}

namespace {
struct SimToSVPass : public circt::impl::LowerSimToSVBase<SimToSVPass> {
  void runOnOperation() override {
    auto circuit = getOperation();
    MLIRContext *context = &getContext();
    LowerDPIFunc lowerDPIFunc(circuit);

    // Lower DPI functions.
    for (auto func :
         llvm::make_early_inc_range(circuit.getOps<sim::DPIFuncOp>()))
      lowerDPIFunc.lower(func);

    std::atomic<bool> usedSynthesisMacro = false;
    auto lowerModule = [&](hw::HWModuleOp module) {
      bool moduleNeeds = moveOpsIntoIfdefGuardsAndProcesses(module);
      module.walk([&](Operation *op) {
        StringRef condName;
        if (auto ifdef = dyn_cast<sv::IfDefOp>(op))
          condName = ifdef.getCond().getName();
        else if (auto ifdefP = dyn_cast<sv::IfDefProceduralOp>(op))
          condName = ifdefP.getCond().getName();

        if (condName == "SYNTHESIS") {
          moduleNeeds = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (moduleNeeds)
        usedSynthesisMacro.store(true);

      SimConversionState state;
      ConversionTarget target(*context);
      target.addIllegalDialect<SimDialect>();
      target.addLegalDialect<sv::SVDialect>();
      target.addLegalDialect<hw::HWDialect>();
      target.addLegalDialect<seq::SeqDialect>();
      target.addLegalDialect<comb::CombDialect>();
      target.addLegalOp<FormatLiteralOp, FormatStringConcatOp, FormatBinOp,
                        FormatDecOp, FormatHexOp, FormatOctOp, FormatCharOp,
                        FormatHierPathOp, FormatScientificOp, FormatFloatOp,
                        FormatGeneralOp>();

      RewritePatternSet patterns(context);
      patterns.add<PlusArgsTestLowering>(context, state);
      patterns.add<PlusArgsValueLowering>(context, state);
      patterns.add<ClockedTerminateOp>(convert);
      patterns.add<ClockedPauseOp>(convert);
      patterns.add<TerminateOp>(convert);
      patterns.add<PauseOp>(convert);
      patterns.add<DPICallLowering>(context, state);
      patterns.add<GetFileLowering>(context, state);
      patterns.add<PrintFormattedProcLowering>(context, state);
      patterns.add<FFlushProcLowering>(context, state);
      patterns.add<ScfIfLowering>(context);
      patterns.add<TimeLowering>(context, state);
      auto result = applyPartialConversion(module, target, std::move(patterns));

      if (failed(result))
        return result;

      SmallVector<Operation *> worklist;
      llvm::SmallDenseSet<Operation *> queued;
      llvm::SmallDenseSet<Operation *> erased;
      auto enqueueFormatDef = [&](Value value) {
        if (auto *defOp = value.getDefiningOp(); defOp && isFormatOp(defOp))
          if (queued.insert(defOp).second)
            worklist.push_back(defOp);
      };
      for (Value root : state.formatCleanupRoots)
        enqueueFormatDef(root);

      while (!worklist.empty()) {
        Operation *candidate = worklist.pop_back_val();
        queued.erase(candidate);
        if (erased.contains(candidate))
          continue;
        if (!candidate->use_empty())
          continue;
        if (auto concat = dyn_cast<FormatStringConcatOp>(candidate))
          for (Value input : concat.getInputs())
            enqueueFormatDef(input);
        erased.insert(candidate);
        candidate->erase();
      }

      // Set the emit fragment.
      lowerDPIFunc.addFragments(module, state.dpiCallees.takeVector());

      if (state.usedSynthesisMacro)
        usedSynthesisMacro = true;
      return result;
    };

    if (failed(mlir::failableParallelForEach(
            context, circuit.getOps<hw::HWModuleOp>(), lowerModule)))
      return signalPassFailure();

    if (usedSynthesisMacro.load()) {
      Operation *op = circuit.lookupSymbol("SYNTHESIS");
      if (op) {
        if (!isa<sv::MacroDeclOp>(op)) {
          op->emitOpError("should be a macro declaration");
          return signalPassFailure();
        }
      } else {
        auto builder = ImplicitLocOpBuilder::atBlockBegin(
            UnknownLoc::get(context), circuit.getBody());
        sv::MacroDeclOp::create(builder, "SYNTHESIS");
      }
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> circt::createLowerSimToSVPass() {
  return std::make_unique<SimToSVPass>();
}
