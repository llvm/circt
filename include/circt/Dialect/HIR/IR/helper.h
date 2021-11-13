#ifndef HIR_HELPER_H
#define HIR_HELPER_H

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"

namespace helper {
llvm::Optional<int64_t> getBitWidth(mlir::Type);
unsigned clog2(int);

bool isBuiltinSizedType(mlir::Type);
bool isBusLikeType(mlir::Type);
llvm::Optional<int64_t> getConstantIntValue(mlir::Value var);
mlir::LogicalResult isConstantIntValue(mlir::Value var);
mlir::IntegerAttr getI64IntegerAttr(mlir::MLIRContext *context, int value);

circt::hir::TimeType getTimeType(mlir::MLIRContext *context);
mlir::ParseResult parseMemrefPortsArray(mlir::DialectAsmParser &,
                                        mlir::ArrayAttr &);
mlir::ParseResult parseIntegerAttr(mlir::IntegerAttr &value,
                                   mlir::StringRef attrName,
                                   mlir::OpAsmParser &parser,
                                   mlir::OperationState &result);
mlir::DictionaryAttr getDictionaryAttr(mlir::MLIRContext *context,
                                       mlir::StringRef name,
                                       mlir::Attribute attr);

mlir::DictionaryAttr getDictionaryAttr(mlir::Builder &builder,
                                       mlir::StringRef name,
                                       mlir::Attribute attr);
mlir::DictionaryAttr getDictionaryAttr(mlir::RewriterBase &builder,
                                       mlir::StringRef name,
                                       mlir::Attribute attr);
llvm::Optional<int64_t> calcLinearIndex(mlir::ArrayRef<mlir::Value> indices,
                                        mlir::ArrayRef<int64_t> dims);

int64_t extractDelayFromDict(mlir::DictionaryAttr dict);
llvm::Optional<mlir::ArrayAttr>
extractMemrefPortsFromDict(mlir::DictionaryAttr dict);
llvm::Optional<int64_t> getRdLatency(mlir::Attribute port);
bool isWrite(mlir::Attribute port);
bool isRead(mlir::Attribute port);
llvm::StringRef extractBusPortFromDict(mlir::DictionaryAttr dict);
llvm::StringRef getInlineAttrName();
void eraseOps(mlir::SmallVectorImpl<mlir::Operation *> &opsToErase);
mlir::Value lookupOrOriginal(mlir::BlockAndValueMapping &mapper,
                             mlir::Value originalValue);
void setNames(mlir::Operation *, mlir::ArrayRef<mlir::StringRef>);

mlir::SmallVector<mlir::Type> getTypes(mlir::ArrayRef<mlir::Value>);
llvm::Optional<mlir::StringRef> getOptionalName(mlir::Operation *operation,
                                                int64_t resultNum);
llvm::Optional<mlir::StringRef> getOptionalName(mlir::Value v);
llvm::Optional<circt::Type> getElementType(circt::Type);
circt::Operation *declareExternalFuncForCall(
    circt::hir::CallOp callOp, circt::SmallVector<circt::StringRef> inputNames,
    circt::SmallVector<circt::StringRef> resultNames = {});
mlir::Value materializeIntegerConstant(mlir::OpBuilder &builder, int value,
                                       int64_t width);
llvm::Optional<mlir::Type> convertToHWType(mlir::Type type);

mlir::Value insertBusSelectLogic(mlir::OpBuilder &builder,
                                 mlir::Value selectBus, mlir::Value trueBus,
                                 mlir::Value falseBus);
mlir::Value insertMultiBusSelectLogic(mlir::OpBuilder &builder,
                                      mlir::Value selectBusT,
                                      mlir::Value trueBusT,
                                      mlir::Value falseBusT);
mlir::Value emitRegisterAlloca(mlir::OpBuilder &builder, mlir::Type elementTy);

mlir::LogicalResult
validatePositiveConstant(mlir::ArrayRef<mlir::Value> indices);
} // namespace helper
#endif
