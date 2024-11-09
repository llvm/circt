//===- EmitRTGAssembly.cpp - RTG Assembly Emitter -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Assembly emitter implementation for the RTG dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Target/EmitRTGAssembly.h"
#include "circt/Dialect/RTG/IR/RTGVisitors.h"
#include "circt/Dialect/RTG/UserElf.h"
#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

using namespace circt;
using namespace rtg;
using namespace EmitRTGAssembly;

#define DEBUG_TYPE "emit-rtg-assembly"

static void printValue(Value val, raw_ostream &stream) {
  if (auto constOp = val.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto integerAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      stream << integerAttr.getValue().getZExtValue();

  // FIXME: Don't look back through the IR for this.
  if (auto resourceOp = val.getDefiningOp<ResourceOpInterface>())
    resourceOp.printAssembly(stream);
  if (auto registerOp = val.getDefiningOp<RegisterOpInterface>())
    stream << registerOp.getRegisterAssembly();
}

static FailureOr<APInt> getBinary(Value val) {
  if (auto labelDeclOp = val.getDefiningOp<LabelDeclOp>())
    return labelDeclOp->emitError(
        "binary representation cannot be computed for labels");

  if (auto constOp = val.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto integerAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return integerAttr.getValue();

  if (auto resourceOp = val.getDefiningOp<ResourceOpInterface>())
    return resourceOp.getBinary();
  if (auto registerOp = val.getDefiningOp<RegisterOpInterface>())
    return registerOp.getClassIndexBinary();

  return failure();
}

namespace {
class EmitRTGToElf : public RTGOpVisitor<EmitRTGToElf, LogicalResult, int> {

  // map for non-moving inserts
  std::map<int, SmallVector<char>> buffers;
  std::map<int, llvm::raw_svector_ostream> streams;

  llvm::raw_ostream &base_stream;
  UserElfInterface *target;

  const EmitRTGAssemblyOptions &options;

  raw_ostream &getStream(int ctx) {
    if (ctx == -1)
      return base_stream;
    auto iter = streams.find(ctx);
    if (iter != streams.end())
      return iter->second;
    auto [iter2, didInsert] = streams.emplace(ctx, std::ref(buffers[ctx]));
    assert(didInsert);
    return iter2->second;
  }

public:
  EmitRTGToElf(llvm::raw_ostream &os, const EmitRTGAssemblyOptions &options)
      : base_stream(os), target(nullptr), options(options) {}

  ~EmitRTGToElf() {
    if (!target)
      return;

    int maxCTX = 0;
    for (auto &[ctx, buffer] : buffers)
      maxCTX = std::max(maxCTX, ctx);

    target->emitElfHeader(base_stream, maxCTX + 1);
    for (int i = 0; i <= maxCTX; i++) {
      target->emitElfContextHeader(base_stream, i);
      if (buffers.count(i))
        base_stream << buffers[i];
      target->emitElfContextFooter(base_stream, i);
    }
    target->emitElfFooter(base_stream);
  }

  using RTGOpVisitor<EmitRTGToElf, LogicalResult, int>::visitOp;

  LogicalResult visitUnhandledOp(Operation *op, int ctx) {
    op->emitError("emitter unknown RTG operation");
    return failure();
  }

  LogicalResult visitResourceOp(ResourceOpInterface op, int ctx) {
    return success();
  }

  LogicalResult visitOp(RenderedContextOp rendered, int ctx) {
    assert(ctx == -1);
    for (auto [index, region] :
         llvm::zip(rendered.getSelectors(), rendered.getRegions())) {
      auto &body = region->getBlocks().front();
      for (Operation &iop : body.getOperations()) {
        if (failed(dispatchOpVisitor(&iop, index)))
          return failure();
      }
    }
    return success();
  }

  LogicalResult visitOp(SequenceOp seq, int ctx) {
    auto &body = seq.getRegion().getBlocks().front();
    for (Operation &iop : body.getOperations()) {
      if (failed(dispatchOpVisitor(&iop, ctx)))
        return failure();
    }
    return success();
  }

  LogicalResult visitOp(TestOp test, int ctx) {
    for (Operation &iop : *test.getBody())
      if (failed(dispatchOpVisitor(&iop, ctx)))
        return failure();

    return success();
  }

  LogicalResult visitOp(LabelDeclOp labeldecl, int ctx) { return success(); }

  LogicalResult visitOp(LabelOp label, int ctx) {
    auto &os = getStream(ctx);
    if (label.getGlobal()) {
      os << ".global ";
      printValue(label.getLabel(), os);
      os << "\n";
    }
    printValue(label.getLabel(), os);
    os << ":\n";
    return success();
  }

  LogicalResult visitInstruction(InstructionOpInterface instr, int ctx) {
    // Ensure only one target
    assert(!target || !dyn_cast<UserElfInterface>(instr->getDialect()) ||
           target == dyn_cast<UserElfInterface>(instr->getDialect()));
    if (!target && dyn_cast<UserElfInterface>(instr->getDialect()))
      target = dyn_cast<UserElfInterface>(instr->getDialect());

    auto &os = getStream(ctx);
    os << llvm::indent(4);
    auto useBinary = llvm::is_contained(options.unsupportedInstructions,
                                        instr->getName().getStringRef());
    if (useBinary)
      os << "\\\\ ";
    instr.printAssembly(os, [&](Value value) { printValue(value, os); });
    os << "\n";
    if (!useBinary)
      return success();
    os << llvm::indent(4);
    SmallVector<APInt> operands;
    for (auto operand : instr->getOperands()) {
      auto res = getBinary(operand);
      if (failed(res))
        return failure();
      operands.push_back(*res);
    }
    SmallVector<char> str;
    instr.getBinary(operands).toString(str, 16, false);
    os << ".word 0x" << str << "\n";
    return success();
  }

  LogicalResult visitExternalOp(Operation *op, int ctx) { return success(); }
};
} // namespace

LogicalResult
EmitRTGAssembly::emitRTGAssembly(Operation *module, llvm::raw_ostream &os,
                                 const EmitRTGAssemblyOptions &options) {
  if (module->getNumRegions() != 1)
    return module->emitError("must have exactly one region");
  if (!module->getRegion(0).hasOneBlock())
    return module->emitError("op region must have exactly one block");

  EmitRTGToElf emitter(os, options);
  for (auto test : module->getRegion(0).getOps<TestOp>())
    if (failed(emitter.dispatchOpVisitor(test, -1)))
      return failure();

  return success();
}

void EmitRTGAssembly::parseUnsupportedInstructionsFile(
    const std::string &unsupportedInstructionsFile,
    SmallVectorImpl<std::string> &unsupportedInstrs) {
  if (!unsupportedInstructionsFile.empty()) {
    std::ifstream input(unsupportedInstructionsFile);
    std::string token;
    while (std::getline(input, token, ','))
      unsupportedInstrs.push_back(token);
  }
}

//===----------------------------------------------------------------------===//
// circt-translate registration
//===----------------------------------------------------------------------===//

// MASSIVE HACK
void (*extraRTGDialects)(mlir::DialectRegistry &) = nullptr;

void EmitRTGAssembly::registerEmitRTGAssemblyTranslation() {
  static llvm::cl::opt<std::string> unsupportedInstructionsFile(
      "emit-assembly-binary-instr-file",
      llvm::cl::desc("File with a comma-separated list of instructions "
                     "not supported by the assembler."),
      llvm::cl::init(""));

  static llvm::cl::list<std::string> unsupportedInstructions(
      "emit-assembly-binary-instr",
      llvm::cl::desc(
          "Comma-separated list of instructions supported by the assembler."),
      llvm::cl::MiscFlags::CommaSeparated);

  static mlir::TranslateFromMLIRRegistration toAssembly(
      "emit-assembly", "emit assembly",
      [=](Operation *moduleOp, raw_ostream &output) {
        EmitRTGAssemblyOptions options;
        SmallVector<std::string> unsupportedInstrs(
            unsupportedInstructions.begin(), unsupportedInstructions.end());
        parseUnsupportedInstructionsFile(unsupportedInstructionsFile.getValue(),
                                         unsupportedInstrs);
        options.unsupportedInstructions = unsupportedInstrs;
        return EmitRTGAssembly::emitRTGAssembly(moduleOp, output, options);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<rtgtest::RTGTestDialect, rtg::RTGDialect,
                        mlir::arith::ArithDialect>();
        if (extraRTGDialects)
          extraRTGDialects(registry);
      });
}
