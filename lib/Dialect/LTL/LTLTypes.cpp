#include "circt/Dialect/LTL/LTLTypes.h"
#include "circt/Dialect/LTL/LTLOps.h"

using namespace circt;
using namespace ltl;

bool circt::ltl::isClocked(mlir::Type type) {
  return isa<ClockedPropertyType>(type) || isa<ClockedSequenceType>(type) ||
         isa<ClockedDisabledPropertyType>(type);
}

bool circt::ltl::isDisabled(mlir::Type type) {
  return isa<DisabledPropertyType>(type) ||
         isa<ClockedDisabledPropertyType>(type);
}

bool circt::ltl::isProperty(mlir::Type type) {
  return isa<ClockedPropertyType>(type) || isa<DisabledPropertyType>(type) ||
         isa<ClockedDisabledPropertyType>(type) || isa<PropertyType>(type);
}

bool circt::ltl::isSequence(mlir::Type type) {
  return isa<SequenceType>(type) || isa<ClockedSequenceType>(type);
}
