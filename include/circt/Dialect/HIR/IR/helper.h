#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/IR/Dialect.h"
// returns the bitwidth of the type.
namespace helper {
unsigned getBitWidth(mlir::Type);
unsigned clog2(int);

/// A primitive type is integer, float or a tuple/tensor of a primitive type.
bool isPrimitiveType(mlir::Type);
unsigned getSizeFromShape(mlir::ArrayRef<int64_t> shape);
mlir::IntegerAttr getIntegerAttr(mlir::MLIRContext *context, int width,
                                 int value);
mlir::IntegerType getIntegerType(mlir::MLIRContext *context, int bitwidth);

mlir::hir::TimeType getTimeType(mlir::MLIRContext *context);

mlir::ParseResult parseIntegerAttr(mlir::IntegerAttr &value, int bitwidth,
                                   mlir::StringRef attrName,
                                   mlir::OpAsmParser &parser,
                                   mlir::OperationState &result);

int64_t getConstantIntValue(mlir::Value var);
int64_t calcLinearIndex(mlir::OperandRange indices,
                        mlir::ArrayRef<int64_t> dims);
} // namespace helper
