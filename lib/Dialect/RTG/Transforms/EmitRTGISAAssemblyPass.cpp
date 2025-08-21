//===- EmitRTGISAAssemblyPass.cpp - RTG Assembly Emitter ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main ISA Assembly emitter implementation for the RTG dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyOpInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/Path.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_EMITRTGISAASSEMBLYPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace circt;
using namespace rtg;

#define DEBUG_TYPE "emit-rtg-isa-assembly"

namespace {

class Emitter {
public:
  Emitter(llvm::raw_ostream &os, const DenseSet<StringAttr> &unsupportedInstr)
      : os(os), unsupportedInstr(unsupportedInstr) {}

  LogicalResult emitFile(emit::FileOp fileOp) {
    auto res = emitBlock(fileOp.getBody());
    state.clear();
    return res;
  }

  LogicalResult emitBlock(Block *block) {
    for (auto &op : *block) {
      if (op.hasTrait<OpTrait::ConstantLike>()) {
        SmallVector<OpFoldResult> results;
        if (failed(op.fold(results)))
          return failure();

        for (auto [val, res] : llvm::zip(op.getResults(), results)) {
          auto attr = res.dyn_cast<Attribute>();
          if (!attr)
            return failure();

          state[val] = attr;
        }

        continue;
      }

      auto res =
          TypeSwitch<Operation *, LogicalResult>(&op)
              .Case<InstructionOpInterface, LabelDeclOp, LabelOp, CommentOp, SpaceOp, SegmentOp>(
                  [&](auto op) { return emit(op); })
              .Default([](auto op) {
                return op->emitError("emitter unknown RTG operation");
              });

      if (failed(res))
        return failure();
    }

    return success();
  }

private:
  LogicalResult emit(InstructionOpInterface instr) {
    os << llvm::indent(4);
    bool useBinary =
        unsupportedInstr.contains(instr->getName().getIdentifier());

    // TODO: we cannot just assume that double-slash is the way to do a line
    // comment
    if (useBinary)
      os << "# ";

    SmallVector<Attribute> operands;
    for (auto operand : instr->getOperands()) {
      if (isa<LabelType>(operand.getType()) && useBinary)
        return instr->emitError("labels cannot be emitted as binary");

      auto attr = state.lookup(operand);
      if (!attr)
        return failure();

      operands.push_back(attr);
    }

    instr.printInstructionAssembly(os, operands);
    os << "\n";

    if (!useBinary)
      return success();

    os << llvm::indent(4);
    // TODO: don't hardcode '.word'
    os << ".word 0x";
    instr.printInstructionBinary(os, operands);
    os << "\n";

    return success();
  }

  LogicalResult emit(LabelDeclOp op) {
    if (!op.getArgs().empty())
      return op->emitError(
          "label arguments must be elaborated before emission");

    state[op.getLabel()] = op.getFormatStringAttr();
    return success();
  }

  LogicalResult emit(LabelOp op) {
    auto labelStr = cast<StringAttr>(state[op.getLabel()]).getValue();
    if (op.getVisibility() == LabelVisibility::external) {
      os << ".extern " << labelStr << "\n";
      return success();
    }

    if (op.getVisibility() == LabelVisibility::global)
      os << ".global " << labelStr << "\n";

    os << labelStr << ":\n";
    return success();
  }

  LogicalResult emit(CommentOp op) {
    os << llvm::indent(4) << "# " << op.getComment() << "\n";
    return success();
  }

  LogicalResult emit(SpaceOp op) {
    os << llvm::indent(4) << ".space " << op.getSize() << "\n";
    return success();
  }

  LogicalResult emit(SegmentOp op) {
    os << "." << op.getKind() << "\n";
    return emitBlock(op.getBody());
  }

private:
  /// Output Stream.
  llvm::raw_ostream &os;

  /// Instructions to emit in binary.
  const DenseSet<StringAttr> &unsupportedInstr;

  /// Evaluated values.
  DenseMap<Value, Attribute> state;
};

} // namespace

static void
parseUnsupportedInstructionsFile(MLIRContext *ctxt,
                                 const std::string &unsupportedInstructionsFile,
                                 DenseSet<StringAttr> &unsupportedInstrs) {
  if (!unsupportedInstructionsFile.empty()) {
    std::ifstream input(unsupportedInstructionsFile, std::ios::in);
    std::string token;
    while (std::getline(input, token, ',')) {
      auto trimmed = StringRef(token).trim();
      if (!trimmed.empty())
        unsupportedInstrs.insert(StringAttr::get(ctxt, trimmed));
    }
  }
}

//===----------------------------------------------------------------------===//
// EmitRTGISAAssemblyPass
//===----------------------------------------------------------------------===//

namespace {
struct EmitRTGISAAssemblyPass
    : public rtg::impl::EmitRTGISAAssemblyPassBase<EmitRTGISAAssemblyPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void EmitRTGISAAssemblyPass::runOnOperation() {
  // Get the set of instructions not supported by the assembler
  DenseSet<StringAttr> unsupportedInstr;
  for (const auto &instr : unsupportedInstructions)
    unsupportedInstr.insert(StringAttr::get(&getContext(), instr));
  parseUnsupportedInstructionsFile(
      &getContext(), unsupportedInstructionsFile.getValue(), unsupportedInstr);

  // Create the output file
  auto filename = getOperation().getFileName();
  std::unique_ptr<llvm::ToolOutputFile> file;
  bool emitToFile = !filename.empty() && filename != "-";
  if (emitToFile) {
    file = createOutputFile(filename, std::string(),
                            [&]() { return getOperation().emitError(); });
    if (!file)
      return signalPassFailure();

    file->keep();
  }

  Emitter emitter(emitToFile ? file->os()
                             : (filename.empty() ? llvm::errs() : llvm::outs()),
                  unsupportedInstr);
  if (failed(emitter.emitFile(getOperation())))
    return signalPassFailure();
}
