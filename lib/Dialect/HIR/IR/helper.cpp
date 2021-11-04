#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"
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

llvm::Optional<uint64_t> getBitWidth(Type type) {
  if (type.dyn_cast<hir::TimeType>())
    return 1;
  if (auto intTy = type.dyn_cast<IntegerType>())
    return intTy.getWidth();
  if (auto floatTy = type.dyn_cast<mlir::FloatType>())
    return floatTy.getWidth();
  if (auto busTy = type.dyn_cast<hir::BusType>())
    return getBitWidth(busTy.getElementType());
  if (auto tupleTy = type.dyn_cast<TupleType>()) {
    int width = 1;
    for (Type ty : tupleTy.getTypes()) {
      auto widthTy = getBitWidth(ty);
      if (!widthTy)
        return llvm::None;
      width *= widthTy.getValue();
    }
    return width;
  }
  if (auto tensorTy = type.dyn_cast<mlir::TensorType>()) {
    int size = 1;
    for (auto szDim : tensorTy.getShape())
      size *= szDim;

    auto widthElementTy = getBitWidth(tensorTy.getElementType());
    if (!widthElementTy)
      return llvm::None;
    return size * widthElementTy.getValue();
  }

  return llvm::None;
}

unsigned clog2(int value) { return (int)(ceil(log2(((double)value)))); }

IntegerAttr getI64IntegerAttr(MLIRContext *context, int value) {
  return IntegerAttr::get(IntegerType::get(context, 64), APInt(64, value));
}

DictionaryAttr getDictionaryAttr(mlir::Builder &builder, StringRef name,
                                 Attribute attr) {
  return DictionaryAttr::get(builder.getContext(),
                             builder.getNamedAttr(name, attr));
}

DictionaryAttr getDictionaryAttr(mlir::MLIRContext *context, StringRef name,
                                 Attribute attr) {
  Builder builder(context);
  return DictionaryAttr::get(context, builder.getNamedAttr(name, attr));
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

bool isBusType(mlir::Type ty) {
  if (ty.isa<hir::BusType>())
    return true;
  if (auto tensorTy = ty.dyn_cast<mlir::TensorType>())
    if (tensorTy.getElementType().isa<hir::BusType>())
      return true;
  return false;
}

TimeType getTimeType(MLIRContext *context) { return TimeType::get(context); }

mlir::ParseResult parseMemrefPortsArray(mlir::DialectAsmParser &parser,
                                        mlir::ArrayAttr &ports) {
  SmallVector<StringRef> portsArray;
  if (parser.parseLParen())
    return failure();

  do {
    StringRef keyword;
    if (succeeded(parser.parseKeyword("send")))
      keyword = "send";
    else if (succeeded(parser.parseKeyword("recv")))
      keyword = "recv";
    else
      return parser.emitError(parser.getCurrentLocation())
             << "Expected 'send' or 'recv' keyword";
    portsArray.push_back(keyword);
  } while (succeeded(parser.parseOptionalComma()));

  ports = parser.getBuilder().getStrArrayAttr(portsArray);
  return success();
}

ParseResult parseIntegerAttr(IntegerAttr &value, StringRef attrName,
                             OpAsmParser &parser, OperationState &result) {

  return parser.parseAttribute(
      value, IntegerType::get(parser.getBuilder().getContext(), 64), attrName,
      result.attributes);
}

llvm::Optional<int64_t> getConstantIntValue(Value var) {
  auto constantOp =
      dyn_cast_or_null<mlir::arith::ConstantOp>(var.getDefiningOp());
  if (!constantOp)
    return llvm::None;

  auto integerAttr = constantOp.value().dyn_cast<IntegerAttr>();
  assert(integerAttr);
  return integerAttr.getInt();
}

mlir::LogicalResult isConstantIntValue(mlir::Value var) {

  if (dyn_cast<mlir::arith::ConstantOp>(var.getDefiningOp()))
    return success();
  return failure();
}

llvm::Optional<int64_t> calcLinearIndex(mlir::ArrayRef<mlir::Value> indices,
                                        mlir::ArrayRef<int64_t> dims) {
  int64_t linearIdx = 0;
  int64_t stride = 1;
  // This can happen if there are no BANK(ADDR) indices.
  if (indices.size() == 0)
    return 0;

  for (int i = indices.size() - 1; i >= 0; i--) {
    auto idxConst = getConstantIntValue(indices[i]);
    if (!idxConst)
      return llvm::None;
    linearIdx += idxConst.getValue() * stride;
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

llvm::Optional<ArrayAttr>
extractMemrefPortsFromDict(mlir::DictionaryAttr dict) {
  if (!dict.getNamed("hir.memref.ports").hasValue())
    return llvm::None;
  return dict.getNamed("hir.memref.ports")
      .getValue()
      .second.dyn_cast<ArrayAttr>();
}

llvm::Optional<uint64_t> getRdLatency(Attribute port) {
  auto portDict = port.dyn_cast<DictionaryAttr>();
  auto rdLatencyAttr = portDict.getNamed("rd_latency");
  if (rdLatencyAttr)
    return rdLatencyAttr.getValue().second.dyn_cast<IntegerAttr>().getInt();
  return llvm::None;
}

bool isWrite(Attribute port) {
  auto portDict = port.dyn_cast<DictionaryAttr>();
  auto wrLatencyAttr = portDict.getNamed("wr_latency");
  if (wrLatencyAttr)
    return true;
  return false;
}

bool isRead(Attribute port) {
  auto portDict = port.dyn_cast<DictionaryAttr>();
  auto rdLatencyAttr = portDict.getNamed("rd_latency");
  if (rdLatencyAttr)
    return true;
  return false;
}

StringRef extractBusPortFromDict(mlir::DictionaryAttr dict) {
  auto ports =
      dict.getNamed("hir.bus.ports").getValue().second.dyn_cast<ArrayAttr>();
  // Bus port should be either send or recv.
  assert(ports.size() == 1);
  return ports[0].dyn_cast<StringAttr>().getValue();
}
llvm::StringRef getInlineAttrName() { return "inline"; }
void eraseOps(mlir::ArrayRef<mlir::Operation *> opsToErase) {
  // Erase the ops in reverse order so that if there are any dependent ops, they
  // get erased first.
  for (auto op = opsToErase.rbegin(); op != opsToErase.rend(); op++)
    (*op)->erase();
}

Value lookupOrOriginal(BlockAndValueMapping &mapper, Value originalValue) {
  if (mapper.contains(originalValue))
    return mapper.lookup(originalValue);
  return originalValue;
}
void setNames(Operation *operation, ArrayRef<StringRef> names) {
  OpBuilder builder(operation);
  operation->setAttr("names", builder.getStrArrayAttr(names));
}
SmallVector<Type> getTypes(ArrayRef<Value> values) {
  SmallVector<Type> types;
  for (auto value : values)
    types.push_back(value.getType());
  return types;
}

llvm::Optional<StringRef> getOptionalName(Operation *operation,
                                          uint64_t resultNum) {
  auto namesAttr = operation->getAttr("names").dyn_cast_or_null<ArrayAttr>();
  if (!namesAttr)
    return llvm::None;
  auto nameAttr = namesAttr[resultNum].dyn_cast_or_null<StringAttr>();
  if (!nameAttr)
    return llvm::None;
  auto name = nameAttr.getValue();
  if (name.size() == 0)
    return llvm::None;
  return name;
}

circt::Type getElementType(circt::Type ty) {
  if (auto tensorTy = ty.dyn_cast<mlir::TensorType>())
    return getElementType(tensorTy.getElementType());
  if (auto busTy = ty.dyn_cast<hir::BusType>())
    return getElementType(busTy.getElementType());
  return ty;
}

Operation *declareExternalFuncForCall(hir::CallOp callOp,
                                      SmallVector<StringRef> inputNames,
                                      SmallVector<StringRef> resultNames) {
  if (callOp.getCalleeDecl())
    return NULL;
  OpBuilder builder(callOp);
  auto moduleOp = callOp->getParentOfType<ModuleOp>();
  builder.setInsertionPointToStart(&moduleOp.body().front());

  auto declOp = builder.create<hir::FuncExternOp>(
      builder.getUnknownLoc(), callOp.calleeAttr().getAttr(),
      TypeAttr::get(callOp.getFuncType()));

  declOp.getFuncBody().push_back(new Block);
  OpBuilder declOpBuilder(declOp);
  FuncExternOp::ensureTerminator(declOp.getFuncBody(), declOpBuilder,
                                 builder.getUnknownLoc());
  // declOp.getFuncBody().front();
  assert(inputNames.size() == callOp.getFuncType().getInputTypes().size());
  inputNames.push_back("t");
  declOp->setAttr("argNames", builder.getStrArrayAttr(inputNames));

  if (resultNames.size() > 0) {
    assert(resultNames.size() == callOp.getFuncType().getResultTypes().size());
    declOp->setAttr("resultNames", builder.getStrArrayAttr(resultNames));
  }
  if (auto params = callOp->getAttr("params"))
    declOp->setAttr("params", params);
  return declOp;
}

Value materializeIntegerConstant(OpBuilder &builder, int value,
                                 uint64_t width) {
  return builder.create<hw::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getIntegerAttr(IntegerType::get(builder.getContext(), width),
                             value));
}

} // namespace helper
