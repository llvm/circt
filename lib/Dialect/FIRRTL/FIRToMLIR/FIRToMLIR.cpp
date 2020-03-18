//===- FIRToMLIR.cpp - .fir to FIRRTL dialect parser ----------------------===//
//
// This implements a .fir file parser.
//
//===----------------------------------------------------------------------===//

#include "spt/Dialect/FIRRTL/FIRToMLIR.h"
#include "FIRLexer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Module.h"
#include "mlir/Translation.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace spt;
using namespace firrtl;
using namespace mlir;
using llvm::SMLoc;
using llvm::SourceMgr;

// Parse the specified .fir file into the specified MLIR context.
OwningModuleRef parseFIRFile(SourceMgr &sourceMgr, MLIRContext *context) {
  auto *dialect = context->getRegisteredDialect<FIRRTLDialect>();
  assert(dialect && "Could not find FIRRTL dialect?");
  (void)dialect;

  auto *buf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  auto loc = SMLoc::getFromPointer(buf->getBufferStart());

  FIRLexer l(sourceMgr, context);

  for (auto tok = l.lexToken(); tok.isNot(FIRToken::eof); tok = l.lexToken()) {
    llvm::outs() << "TOKEN: '" << tok.getSpelling() << "'\n";
  }

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
