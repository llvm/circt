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
/// This struct represents a time-instant at timeVar+offset.
struct TimeInstant {
  Value timeVar;
  uint64_t offset;
};

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
  bool isAlwaysValid(Value);
  Value getRootTimeVar(Value);
  uint64_t getRootTimeOffset(Value);

private:
  ScheduleInfo(FuncOp op) : funcOp(op) {}

public:
  llvm::DenseMap<Value, Value> mapValueToRootTimeVar;
  llvm::DenseMap<Value, uint64_t> mapValueToOffset;
  llvm::SetVector<Value> setOfAlwaysValidValues;
  llvm::SetVector<Value> setOfRootTimeVars;
  hir::FuncOp funcOp;
};

} // namespace hir
} // namespace circt
