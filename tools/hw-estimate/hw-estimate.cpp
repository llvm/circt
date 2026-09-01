/**
 * @file hw-estimate.cpp
 * @author Terrence Cao
 * @brief Lightweight tool to estimate the GE and FO4 of a potential hardware design
 * @details current usage: build/bin/circt-verilog <verilog file> | build/bin/hw-estimate -
 *
 */



#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFileName(
    llvm::cl::Positional, llvm::cl::desc("<input .mlir file>"),
    llvm::cl::init("-")
);

int main(int argc, char** argv)
{
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "hw-estimate\n");

  DialectRegistry registry;
  registry.insert<circt::hw::HWDialect, circt::comb::CombDialect, circt::seq::SeqDialect>();

  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(inputFileName, &context);

  if(!module)
  {
    llvm::errs() << "Failed to parse the input\n";
    return 1;
  }

  module->walk([&](Operation *op)
  {
    llvm::outs() << op->getName().getStringRef() << "\n";
  });

  return 0;
}
