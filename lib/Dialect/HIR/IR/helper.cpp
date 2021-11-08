#include "circt/Dialect/Comb/CombOps.h"
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

llvm::Optional<int64_t> getBitWidth(Type type) {
  if (type.dyn_cast<hir::TimeType>())
    return 1;

  if (auto intTy = type.dyn_cast<IntegerType>())
    return intTy.getWidth();

  if (auto floatTy = type.dyn_cast<mlir::FloatType>())
    return floatTy.getWidth();

  if (auto tensorTy = type.dyn_cast<mlir::TensorType>())
    return tensorTy.getNumElements() *
           getBitWidth(tensorTy.getElementType()).getValue();

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

  if (auto busTy = type.dyn_cast<hir::BusType>())
    return getBitWidth(busTy.getElementType());

  if (auto busTensorTy = type.dyn_cast<hir::BusTensorType>())
    return busTensorTy.getNumElements() *
           getBitWidth(busTensorTy.getElementType()).getValue();

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

bool isBusLikeType(mlir::Type ty) {
  return (ty.isa<hir::BusType>() || ty.isa<hir::BusTensorType>());
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

llvm::Optional<int64_t> getRdLatency(Attribute port) {
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
                                          int64_t resultNum) {
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

llvm::Optional<mlir::StringRef> getOptionalName(mlir::Value v) {
  Operation *operation = v.getDefiningOp();
  if (operation) {
    for (size_t i = 0; i < operation->getNumResults(); i++) {
      if (operation->getResult(i) == v) {
        return getOptionalName(operation, i);
      }
    }
    return llvm::None;
  }
  auto *bb = v.getParentBlock();
  auto argNames = bb->getParentOp()->getAttrOfType<mlir::ArrayAttr>("argNames");
  if (argNames) {
    assert(argNames.size() == bb->getNumArguments());
    for (size_t i = 0; i < bb->getNumArguments(); i++) {
      if (v == bb->getArgument(i)) {
        return argNames[i].dyn_cast<mlir::StringAttr>().getValue();
      }
    }
  }
  return llvm::None;
}

llvm::Optional<Type> getElementType(circt::Type ty) {
  if (auto tensorTy = ty.dyn_cast<mlir::TensorType>())
    return tensorTy.getElementType();
  if (auto busTy = ty.dyn_cast<hir::BusType>())
    return busTy.getElementType();
  if (auto busTensorTy = ty.dyn_cast<hir::BusTensorType>())
    return busTensorTy.getElementType();
  return llvm::None;
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

Value materializeIntegerConstant(OpBuilder &builder, int value, int64_t width) {
  return builder.create<hw::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getIntegerAttr(IntegerType::get(builder.getContext(), width),
                             value));
}

static Optional<Type> convertBusType(hir::BusType busTy) {
  return convertToHWType(busTy.getElementType());
}

static Optional<Type> convertTensorType(mlir::TensorType tensorTy) {
  auto elementHWTy = convertToHWType(tensorTy.getElementType());
  // Tensor element must always be a value type and hence always convertible
  // to a valid hw type.
  if (tensorTy.getNumElements() == 1)
    return elementHWTy;
  return hw::ArrayType::get(*elementHWTy, tensorTy.getNumElements());
}

static Optional<Type> convertBusTensorType(hir::BusTensorType busTensorTy) {
  auto elementHWTy = convertToHWType(busTensorTy.getElementType());
  if (!elementHWTy) {
    Builder builder(busTensorTy.getContext());
    mlir::emitError(builder.getUnknownLoc())
        << "Could not convert bus tensor element type "
        << busTensorTy.getElementType();
  }

  // BusTensor element must always be a value type and hence always convertible
  // to a valid hw type.
  if (busTensorTy.getNumElements() == 1)
    return elementHWTy;
  return hw::ArrayType::get(*elementHWTy, busTensorTy.getNumElements());
}

static Optional<Type> convertTupleType(mlir::TupleType tupleTy) {
  int64_t width = 0;
  for (auto elementTy : tupleTy.getTypes()) {
    // We can't handle tensors/arrays inside tuple.
    auto elementHWTy = convertToHWType(elementTy);
    if (!(elementHWTy && elementHWTy.getValue().isa<IntegerType>()))
      return llvm::None;
    width += (*elementHWTy).dyn_cast<IntegerType>().getWidth();
  }
  return IntegerType::get(tupleTy.getContext(), width);
}

llvm::Optional<Type> convertToHWType(Type type) {
  if (type.isa<TimeType>())
    return IntegerType::get(type.getContext(), 1);
  if (type.isa<IntegerType>())
    return type;
  if (type.isa<mlir::FloatType>())
    return IntegerType::get(type.getContext(), type.getIntOrFloatBitWidth());
  if (auto ty = type.dyn_cast<mlir::TensorType>())
    return convertTensorType(ty);
  if (auto ty = type.dyn_cast<mlir::TupleType>())
    return convertTupleType(ty);
  if (auto ty = type.dyn_cast<hir::BusType>())
    return convertBusType(ty);
  if (auto ty = type.dyn_cast<hir::BusTensorType>())
    return convertBusTensorType(ty);
  return llvm::None;
}

Value insertBusSelectLogic(OpBuilder &builder, Value selectBus, Value trueBus,
                           Value falseBus) {
  auto uLoc = builder.getUnknownLoc();
  return builder
      .create<hir::BusMapOp>(
          builder.getUnknownLoc(),
          ArrayRef<Value>({selectBus, trueBus, falseBus}),
          [&uLoc](OpBuilder &builder, ArrayRef<Value> operands) {
            Value result = builder.create<comb::MuxOp>(
                uLoc, operands[0], operands[1], operands[2]);

            return builder.create<hir::YieldOp>(uLoc, result);
          })
      .getResult(0);
}

Value insertMultiBusSelectLogic(OpBuilder &builder, Value selectBusT,
                                Value trueBusT, Value falseBusT) {
  auto uLoc = builder.getUnknownLoc();
  return builder
      .create<hir::BusTensorMapOp>(
          builder.getUnknownLoc(),
          ArrayRef<Value>({selectBusT, trueBusT, falseBusT}),
          [&uLoc](OpBuilder &builder, ArrayRef<Value> operands) {
            Value result = builder.create<comb::MuxOp>(
                uLoc, operands[0], operands[1], operands[2]);

            return builder.create<hir::YieldOp>(uLoc, result);
          })
      .getResult(0);
}

} // namespace helper
