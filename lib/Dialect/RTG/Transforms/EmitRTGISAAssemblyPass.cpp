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

#include "circt/Dialect/RTG/IR/RTGISAAssemblyOpInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/Path.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
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

  LogicalResult emitTest(rtg::TestOp test, bool emitHeaderFooter = false) {
    if (emitHeaderFooter)
      os << "# Begin of " << test.getSymName() << "\n\n";

    for (auto &op : *test.getBody()) {
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

      auto res = TypeSwitch<Operation *, LogicalResult>(&op)
                     .Case<InstructionOpInterface, LabelDeclOp, LabelOp>(
                         [&](auto op) { return emit(op); })
                     .Default([](auto op) {
                       return op->emitError("emitter unknown RTG operation");
                     });

      if (failed(res))
        return failure();
    }

    state.clear();

    if (emitHeaderFooter)
      os << "\n# End of " << test.getSymName() << "\n\n";

    return success();
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
    std::ifstream input(unsupportedInstructionsFile);
    std::string token;
    while (std::getline(input, token, ',')) {
      auto trimmed = StringRef(token).trim();
      if (!trimmed.empty())
        unsupportedInstrs.insert(StringAttr::get(ctxt, trimmed));
    }
  }
}

static std::unique_ptr<llvm::ToolOutputFile>
createOutputFile(StringRef filename, StringRef dirname,
                 function_ref<InFlightDiagnostic()> emitError) {
  // Determine the output path from the output directory and filename.
  SmallString<128> outputFilename(dirname);
  appendPossiblyAbsolutePath(outputFilename, filename);
  auto outputDir = llvm::sys::path::parent_path(outputFilename);

  // Create the output directory if needed.
  std::error_code error = llvm::sys::fs::create_directories(outputDir);
  if (error) {
    emitError() << "cannot create output directory \"" << outputDir
                << "\": " << error.message();
    return {};
  }

  // Open the output file.
  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output)
    emitError() << errorMessage;
  return output;
}

//===----------------------------------------------------------------------===//
// EmitRTGISAAssemblyPass
//===----------------------------------------------------------------------===//

namespace {
struct EmitRTGISAAssemblyPass
    : public rtg::impl::EmitRTGISAAssemblyPassBase<EmitRTGISAAssemblyPass> {
  using Base::Base;

  void runOnOperation() override;
  /// Emit each 'rtg.test' into a separate file using the test's name as the
  /// filename.
  LogicalResult emitSplit(const DenseSet<StringAttr> &unsupportedInstr);
  /// Emit all tests into a single file (or print them to stderr if no file path
  /// is given).
  LogicalResult emit(const DenseSet<StringAttr> &unsupportedInstr);
};
} // namespace

void EmitRTGISAAssemblyPass::runOnOperation() {
  if ((!path.hasValue() || path.empty()) && splitOutput) {
    getOperation().emitError("'split-output' option only valid in combination "
                             "with a valid 'path' argument");
    return signalPassFailure();
  }

  DenseSet<StringAttr> unsupportedInstr;
  for (const auto &instr : unsupportedInstructions)
    unsupportedInstr.insert(StringAttr::get(&getContext(), instr));
  parseUnsupportedInstructionsFile(
      &getContext(), unsupportedInstructionsFile.getValue(), unsupportedInstr);

  if (splitOutput) {
    if (failed(emitSplit(unsupportedInstr)))
      return signalPassFailure();

    return;
  }

  if (failed(emit(unsupportedInstr)))
    return signalPassFailure();
}

LogicalResult
EmitRTGISAAssemblyPass::emit(const DenseSet<StringAttr> &unsupportedInstr) {
  std::unique_ptr<llvm::ToolOutputFile> file;
  bool emitToFile = path.hasValue() && !path.empty() && path != "-";
  if (emitToFile) {
    file = createOutputFile(path, std::string(),
                            [&]() { return getOperation().emitError(); });
    if (!file)
      return failure();

    file->keep();
  }

  Emitter emitter(emitToFile ? file->os()
                             : (path == "-" ? llvm::outs() : llvm::errs()),
                  unsupportedInstr);
  for (auto test : getOperation().getOps<TestOp>())
    if (failed(emitter.emitTest(test, true)))
      return failure();

  return success();
}

LogicalResult EmitRTGISAAssemblyPass::emitSplit(
    const DenseSet<StringAttr> &unsupportedInstr) {
  auto tests = getOperation().getOps<TestOp>();
  return failableParallelForEach(
      &getContext(), tests.begin(), tests.end(), [&](rtg::TestOp test) {
        auto res = createOutputFile(test.getSymName().str() + ".s", path,
                                    [&]() { return test.emitError(); });
        if (!res)
          return failure();

        res->keep();
        return Emitter(res->os(), unsupportedInstr).emitTest(test);
      });
}
