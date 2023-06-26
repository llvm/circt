#include "circt/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"
#include "circt/Dialect/SV/SVOps.h"
namespace circt {
struct LoopReroller {
  LoopReroller(ImplicitLocOpBuilder &builder,
               unsigned upperLimitTermSize = 8192)
      : builder(builder), upperLimitTermSize(upperLimitTermSize) {}
  mlir::Value unifyTwoValuesImpl(Value value, Value next);
  LogicalResult unifyTwoValues(Value value, Value next);
  LogicalResult unifyIntoTemplateImpl(Value value, Value next);
  LogicalResult unifyIntoTemplate(Value value);
  auto& getOperations() {
    return sandbox.getThenBlock()->getOperations();
  }
  Value getTemplateValue() {
    return templateValue;
  }
  ~LoopReroller(){
    if(sandbox){
    sandbox.erase();
    }
  }
  auto& getDummyValues() {
    return dummyValues;
  }
  unsigned getTermSize() const {
    return termSize;
  }

private:
  ImplicitLocOpBuilder &builder;
  unsigned upperLimitTermSize;
  unsigned termSize = 0;
  llvm::MapVector<Value, SmallVector<Value, 4>> dummyValues;
  mlir::Value templateValue;
  sv::IfDefProceduralOp sandbox;
};

} // namespace circt
