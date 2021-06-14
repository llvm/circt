#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/helper.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include <string>

using namespace mlir;
using namespace hir;
using namespace llvm;

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
  if (auto tupleTy = type.dyn_cast<TupleType>()) {
    int width = 1;
    for (Type ty : tupleTy.getTypes()) {
      width *= getBitWidth(ty);
    }
    return width;
  }
  if (auto tensorTy = type.dyn_cast<TensorType>()) {
    int size = 1;
    for (auto szDim : tensorTy.getShape()) {
      size *= szDim;
    }
    return size * getBitWidth(tensorTy.getElementType());
  }

  // error
  fprintf(stderr, "\nERROR: Can't calculate getBitWidth for type %s.\n",
          typeString(type).c_str());
  assert(false);
  return 0;
}

unsigned clog2(int value) { return (int)(ceil(log2(((double)value)))); }

IntegerAttr getIntegerAttr(MLIRContext *context, int width, int value) {
  return IntegerAttr::get(IntegerType::get(context, width),
                          APInt(width, value));
}

bool isPrimitiveType(Type ty) {
  if (ty.isa<IntegerType>() || ty.isa<FloatType>())
    return true;
  if (ty.isa<TupleType>()) {
    bool tupleMembersArePrimitive = true;
    for (auto memberTy : ty.dyn_cast<TupleType>().getTypes())
      tupleMembersArePrimitive &= isPrimitiveType(memberTy);
    if (tupleMembersArePrimitive)
      return true;
  }
  if (ty.isa<TensorType>() &&
      isPrimitiveType(ty.dyn_cast<TensorType>().getElementType()))
    return true;
  return false;
}

unsigned getSizeFromShape(ArrayRef<int64_t> shape) {
  if (shape.size() == 0)
    return 0;

  unsigned size = 1;
  for (auto dimSize : shape) {
    size *= dimSize;
  }
  return size;
}

IntegerType getIntegerType(MLIRContext *context, int bitwidth) {
  return IntegerType::get(context, bitwidth);
}

TimeType getTimeType(MLIRContext *context) { return TimeType::get(context); }

ParseResult parseIntegerAttr(IntegerAttr &value, int bitwidth,
                             StringRef attrName, OpAsmParser &parser,
                             OperationState &result) {

  return parser.parseAttribute(
      value, getIntegerType(parser.getBuilder().getContext(), bitwidth),
      attrName, result.attributes);
}
} // namespace helper
