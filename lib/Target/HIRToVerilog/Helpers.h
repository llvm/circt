#ifndef __HIRToVerilogHELPERS.H__
#define __HIRToVerilogHELPERS .H__
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Target/HIRToVerilog/HIRToVerilog.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include <locale>
#include <string>

using namespace mlir;
using namespace hir;
using namespace std;

static unsigned max(unsigned x, unsigned y) { return x > y ? x : y; }
static bool isTerminatingChar(char c) {
  if (isalnum(c))
    return false;
  else if (c == '_')
    return false;
  return true;
}

static void findAndReplaceAll(string &data, string toSearch,
                              string replaceStr) {
  int i = 0;
  for (char c : toSearch) {
    if (i == 0 && c == '$')
      continue;
    i++;
    if (isTerminatingChar(c)) {
      string error = "ERROR: toSearch( = \"" + toSearch +
                     "\") can't contain any terminating characters. Found (" +
                     c + ")";
      printf("/*%s*/\n", error.c_str());
      fflush(stdout);
      assert(!isTerminatingChar(c));
    }
  }
  // Get the first occurrence.
  size_t pos = data.find(toSearch);
  // Repeat till end is reached.
  while (pos != string::npos) {
    // Replace this occurrence of Sub String only if its a complete word.
    if (isTerminatingChar(data[pos + toSearch.size()]))
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
