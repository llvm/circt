#include "circt/Dialect/LTL/LTLTypes.h"
#include "circt/Dialect/LTL/LTLOps.h"

using namespace circt;
using namespace ltl;

bool circt::ltl::isClocked(mlir::Type type) {
  return isa<ClockedPropertyType, ClockedSequenceType,
    ClockedDisabledPropertyType>(type);
}

bool circt::ltl::isDisabled(mlir::Type type) {
  return isa<DisabledPropertyType, ClockedDisabledPropertyType>(type);
}

bool circt::ltl::isProperty(mlir::Type type) {
  return isa<
    ClockedPropertyType, DisabledPropertyType,
    ClockedDisabledPropertyType, PropertyType>(type);
}

bool circt::ltl::isSequence(mlir::Type type) {
  return isa<SequenceType, ClockedSequenceType>(type);
}
