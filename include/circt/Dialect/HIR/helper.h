#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
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
mlir::Type getIntegerType(mlir::MLIRContext *context, int bitwidth);

mlir::hir::ConstType getConstIntType(mlir::MLIRContext *context);

mlir::Type getTimeType(mlir::MLIRContext *context);

mlir::ParseResult parseIntegerAttr(mlir::IntegerAttr &value, int bitwidth,
                                   mlir::StringRef attrName,
                                   mlir::OpAsmParser &parser,
                                   mlir::OperationState &result);

} // namespace helper
