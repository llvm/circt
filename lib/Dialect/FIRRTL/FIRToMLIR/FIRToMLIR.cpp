//===- FIRToMLIR.cpp - .fir to FIRRTL dialect parser ----------------------===//
//
// This implements a .fir file parser.
//
//===----------------------------------------------------------------------===//

#include "spt/Dialect/FIRRTL/FIRToMLIR.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Module.h"
#include "mlir/Translation.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "llvm/Support/SourceMgr.h"
using namespace spt;
using namespace firrtl;
using namespace mlir;
using llvm::SMLoc;
using llvm::SourceMgr;

namespace {
class Lexer {
public:
  Lexer(const llvm::SourceMgr &sourceMgr, MLIRContext *context)
      : sourceMgr(sourceMgr), context(context) {}
  Location getEncodedSourceLocation(llvm::SMLoc loc);

private:
  const llvm::SourceMgr &sourceMgr;
  MLIRContext *context;
};
} // end anonymous namespace

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
Location Lexer::getEncodedSourceLocation(llvm::SMLoc loc) {
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  auto *buffer = sourceMgr.getMemoryBuffer(mainFileID);

  return FileLineColLoc::get(buffer->getBufferIdentifier(), lineAndColumn.first,
                             lineAndColumn.second, context);
}

// Parse the specified .fir file into the specified MLIR context.
OwningModuleRef parseFIRFile(SourceMgr &sourceMgr, MLIRContext *context) {
  auto *dialect = context->getRegisteredDialect<FIRRTLDialect>();
  assert(dialect && "Could not find FIRRTL dialect?");
  (void)dialect;

  auto *buf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  auto loc = SMLoc::getFromPointer(buf->getBufferStart());

  Lexer l(sourceMgr, context);
  mlir::emitError(l.getEncodedSourceLocation(loc),
                  "firrtl parsing not implemented");

  return {};
}

void spt::firrtl::registerFIRRTLToMLIRTranslation() {
  static TranslateToMLIRRegistration fromLLVM(
      "import-firrtl", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        return parseFIRFile(sourceMgr, context);
      });
}
