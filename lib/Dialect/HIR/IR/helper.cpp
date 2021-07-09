#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include <string>

using namespace circt;
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
  if (auto floatTy = type.dyn_cast<mlir::FloatType>())
    return floatTy.getWidth();
  if (auto tupleTy = type.dyn_cast<TupleType>()) {
    int width = 1;
    for (Type ty : tupleTy.getTypes()) {
      width *= getBitWidth(ty);
    }
    return width;
  }
  if (auto tensorTy = type.dyn_cast<mlir::TensorType>()) {
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

IntegerAttr getIntegerAttr(MLIRContext *context, int value) {
  return IntegerAttr::get(IntegerType::get(context, 64), APInt(64, value));
}

DictionaryAttr getDictionaryAttr(mlir::Builder &builder, StringRef name,
                                 Attribute attr) {
  return DictionaryAttr::get(builder.getContext(),
                             builder.getNamedAttr(name, attr));
}

DictionaryAttr getDictionaryAttr(mlir::RewriterBase &rewriter, StringRef name,
                                 Attribute attr) {
  return DictionaryAttr::get(rewriter.getContext(),
                             rewriter.getNamedAttr(name, attr));
}

bool isBuiltinSizedType(Type ty) {
  if ((ty.isa<IntegerType>() && ty.dyn_cast<IntegerType>().isSignless()) ||
      ty.isa<mlir::FloatType>())
    return true;
  if (ty.isa<TupleType>()) {
    bool tupleMembersArePrimitive = true;
    for (auto memberTy : ty.dyn_cast<TupleType>().getTypes())
      tupleMembersArePrimitive &= isBuiltinSizedType(memberTy);
    if (tupleMembersArePrimitive)
      return true;
  }
  if (ty.isa<mlir::TensorType>() &&
      isBuiltinSizedType(ty.dyn_cast<mlir::TensorType>().getElementType()))
    return true;
  return false;
}

TimeType getTimeType(MLIRContext *context) { return TimeType::get(context); }

ParseResult parseIntegerAttr(IntegerAttr &value, StringRef attrName,
                             OpAsmParser &parser, OperationState &result) {

  return parser.parseAttribute(
      value, IntegerType::get(parser.getBuilder().getContext(), 64), attrName,
      result.attributes);
}

int64_t getConstantIntValue(Value var) {
  auto constantOp = dyn_cast<mlir::ConstantOp>(var.getDefiningOp());
  assert(constantOp);
  auto integerAttr = constantOp.value().dyn_cast<IntegerAttr>();
  assert(integerAttr);
  return integerAttr.getInt();
}

mlir::LogicalResult isConstantIntValue(mlir::Value var) {

  if (dyn_cast<mlir::ConstantOp>(var.getDefiningOp()))
    return success();
  return failure();
}

int64_t calcLinearIndex(mlir::ArrayRef<mlir::Value> indices,
                        mlir::ArrayRef<int64_t> dims) {
  int64_t linearIdx = 0;
  int64_t stride = 1;
  assert(indices.size() != 0);

  for (int i = indices.size() - 1; i >= 0; i--) {
    linearIdx += getConstantIntValue(indices[i]) * stride;
    stride *= dims[i];
  }
  return linearIdx;
}

int64_t extractDelayFromDict(mlir::DictionaryAttr dict) {
  return dict.getNamed("hir.delay")
      .getValue()
      .second.dyn_cast<IntegerAttr>()
      .getInt();
}

ArrayAttr extractMemrefPortsFromDict(mlir::DictionaryAttr dict) {
  return dict.getNamed("hir.memref.ports")
      .getValue()
      .second.dyn_cast<ArrayAttr>();
}

StringRef extractBusPortFromDict(mlir::DictionaryAttr dict) {
  auto ports =
      dict.getNamed("hir.bus.ports").getValue().second.dyn_cast<ArrayAttr>();
  // Bus port should be either send or recv.
  assert(ports.size() == 1);
  return ports[0].dyn_cast<StringAttr>().getValue();
}
} // namespace helper
