#include "circt/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"
#include "circt/Dialect/SV/SVOps.h"
// TODO: Make this iterative.
namespace circt {
struct LoopReroller {
  LoopReroller(ImplicitLocOpBuilder &builder,
               unsigned upperLimitTermSize = 8192)
      : builder(builder), upperLimitTermSize(upperLimitTermSize) {}
  mlir::Value unifyTwoValuesImpl(Value value, Value next);
  LogicalResult unifyTwoValues(Value value, Value next);
  LogicalResult unifyIntoTemplateImpl(Value value, Value next);
  LogicalResult unifyIntoTemplate(Value value);

private:
  unsigned upperLimitTermSize;
  llvm::MapVector<Value, SmallVector<Value, 4>> dummyValues;
  mlir::Value templateValue;
  ImplicitLocOpBuilder &builder;
  sv::IfDefProceduralOp sandbox;
};

} // namespace circt
