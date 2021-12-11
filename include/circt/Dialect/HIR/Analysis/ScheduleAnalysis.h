//=========- ScheduleAnalysis.cpp - Generate schedule info ----------------===//
//
// This file defines the ScheduleInfo analysis class. This class holds the
// scheduling info.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace circt {
namespace hir {

/// This class builds the scheduling info for each operation.
/// The schedule-info contains
///   - List of all root time-vars (which can not be expressed as a fixed offset
///   from another time-var).
///   - Mapping from a Value to a time instant. The value is valid at that time
///   instant.
class ScheduleInfo {
public:
  static llvm::Optional<ScheduleInfo> createScheduleInfo(FuncOp);

public:
  bool isValidAtTime(Value v, Value tstart, int64_t offset);
  Value getRootTimeVar(Value);
  int64_t getRootTimeOffset(Value);
  void mapValueToTime(Value, Value, int64_t);
  void mapValueToAlwaysValid(Value);
  void setAsRootTimeVar(Value);
  void setAsAlwaysValidValue(Value);

private:
  ScheduleInfo(FuncOp op) : funcOp(op) {}

  llvm::DenseMap<Value, Value> mapValueToRootTimeVar;
  llvm::DenseMap<Value, int64_t> mapValueToOffset;
  llvm::SetVector<Value> setOfAlwaysValidValues;
  llvm::SetVector<Value> setOfRootTimeVars;
  hir::FuncOp funcOp;
};

} // namespace hir
} // namespace circt
