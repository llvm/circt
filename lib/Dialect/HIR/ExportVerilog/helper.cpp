#include "circt/Dialect/HIR/IR/helper.h"
#include "helper.h"

static bool isTerminatingChar(char c) {
  if (isalnum(c))
    return false;
  if (c == '_')
    return false;
  return true;
}

namespace helper {
void findAndReplaceAll(string &data, string toSearch, string replaceStr) {
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

unsigned calcAddrWidth(hir::MemrefType memrefTy) {
  // FIXME: Currently we assume that all dims are power of two.
  auto shape = memrefTy.getShape();
  auto dimKinds = memrefTy.getDimKinds();
  unsigned addrWidth = 0;
  for (size_t i = 0; i < shape.size(); i++) {
    if (dimKinds[i] == ADDR) {
      addrWidth += helper::clog2(shape[i]);
    }
  }
  return addrWidth;
}
} // namespace helper
