#ifndef __HIRToVerilogHELPERS.H__
#define __HIRToVerilogHELPERS .H__
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Target/HIRToVerilog/HIRToVerilog.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include <string>

using namespace mlir;
using namespace hir;
using namespace std;

static unsigned max(unsigned x, unsigned y) { return x > y ? x : y; }
static void findAndReplaceAll(string &data, string toSearch,
                              string replaceStr) {
  // Get the first occurrence.
  size_t pos = data.find(toSearch);
  // Repeat till end is reached.
  while (pos != string::npos) {
    // Replace this occurrence of Sub String.
    data.replace(pos, toSearch.size(), replaceStr);
    // Get the next occurrence from the current position.
    pos = data.find(toSearch, pos + replaceStr.size());
  }
}

static unsigned getBitWidth(Type ty) {
  unsigned bitwidth = 0;
  if (auto intTy = ty.dyn_cast<IntegerType>()) {
    bitwidth = intTy.getWidth();
  } else if (auto memrefTy = ty.dyn_cast<MemrefType>()) {
    bitwidth = getBitWidth(memrefTy.getElementType());
  } else if (auto constTy = ty.dyn_cast<ConstType>()) {
    bitwidth = getBitWidth(constTy.getElementType());
  }
  return bitwidth;
}

static unsigned calcAddrWidth(hir::MemrefType memrefTy) {
  // FIXME: Currently we assume that all dims are power of two.
  auto shape = memrefTy.getShape();
  auto elementType = memrefTy.getElementType();
  auto packing = memrefTy.getPacking();
  unsigned elementWidth = getBitWidth(elementType);
  int max_dim = shape.size() - 1;
  unsigned addrWidth = 0;
  for (auto dim : packing) {
    // dim0 is last in shape.
    int dim_size = shape[max_dim - dim];
    float log_size = log2(dim_size);
    addrWidth += ceil(log_size);
  }
  return addrWidth;
}
#endif
