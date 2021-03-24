#ifndef __HIRToVerilogHELPER__
#define __HIRToVerilogHELPER__
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Translation/HIRToVerilog.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/Support/raw_ostream.h"

#include <locale>
#include <string>

using namespace mlir;
using namespace hir;
using namespace std;

namespace helper {
void findAndReplaceAll(string &data, string toSearch, string replaceStr);
unsigned getBitWidth(Type type);
unsigned calcAddrWidth(hir::MemrefType memrefTy);
} // namespace helper
#endif
