#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/helper.h"
#include <string>

using namespace mlir;

namespace helper {
std::string typeString(Type t) {
  std::string typeStr;
  llvm::raw_string_ostream typeOstream(typeStr);
  t.print(typeOstream);
  return typeStr;
}

unsigned getBitWidth(Type type) {
  if (type.dyn_cast<hir::TimeType>())
    return 1;
  if (auto intTy = type.dyn_cast<IntegerType>())
    return intTy.getWidth();
  if (auto floatTy = type.dyn_cast<FloatType>())
    return floatTy.getWidth();

  // error
  fprintf(stderr, "\nERROR: Can't calculate getBitWidth for type %s.\n",
          typeString(type).c_str());
  assert(false);
  return 0;
}
} // namespace helper
