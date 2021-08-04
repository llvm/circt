#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
// returns the bitwidth of the type.
namespace helper {
llvm::Optional<uint64_t> getBitWidth(mlir::Type);
unsigned clog2(int);

/// A primitive type is integer, float or a tuple/tensor of a primitive type.
bool isBuiltinSizedType(mlir::Type);
bool isBusType(mlir::Type);
int64_t getConstantIntValue(mlir::Value var);
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
int64_t calcLinearIndex(mlir::ArrayRef<mlir::Value> indices,
                        mlir::ArrayRef<int64_t> dims);

int64_t extractDelayFromDict(mlir::DictionaryAttr dict);
llvm::Optional<mlir::ArrayAttr>
extractMemrefPortsFromDict(mlir::DictionaryAttr dict);
llvm::Optional<uint64_t> getRdLatency(mlir::Attribute port);
llvm::StringRef extractBusPortFromDict(mlir::DictionaryAttr dict);
} // namespace helper
