//===- DeleteLocalVar.cpp - Delete local temporary variables --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DeleteLocalVar pass.
// Use to collect net/variable declarations and bound a value to them.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace moore;

namespace {
struct DeleteLocalVarPass : public DeleteLocalVarBase<DeleteLocalVarPass> {
  void runOnOperation() override;
};
} // namespace

namespace {
struct DLV { // Delete Local Variables
  /// All local temporary variables at the `moore.procedure`, `scf.if`,
  /// `scf.while` body region/block need to be deleted, they won't be reserved
  /// at the low level.
  void deleteLocalVar(DenseSet<Operation *> &localVars,
                      DenseSet<Operation *> &users, mlir::Region &region);

  /// The `valueSymbols` used to collect nets/variables and their initial or
  /// assignment values.
  using ValueSymbols = llvm::ScopedHashTable<Operation *, Value>;
  using ValueSymbolScope = ValueSymbols::ScopeTy;
  ValueSymbols valueSymbols;
};
} // namespace

void DLV::deleteLocalVar(DenseSet<Operation *> &localVars,
                         DenseSet<Operation *> &users, mlir::Region &region) {
  ValueSymbolScope scope(valueSymbols);

  for (auto &op : region.getOps()) {
    // Assume not to assign a new value to the local variable.
    bool isNewValue = false;

    TypeSwitch<Operation *, void>(&op)
        // Collect all local variables and their initial values
        // if exist and all users.
        .Case<VariableOp>([&](auto op) {
          valueSymbols.insert(op, op.getInitial());
          localVars.insert(op);
          for (auto *user : op->getUsers())
            users.insert(user);
        })
        // Update the values of local variables.
        .Case<BlockingAssignOp, NonBlockingAssignOp>([&](auto op) {
          auto destOp = op.getDst().getDefiningOp();
          auto srcOp = op.getSrc().getDefiningOp();

          // For example, `a = 1`, `b = a`, `c = b` both are local variable.
          // After updating: `a = 1`, `b = 1`, `c = 1`.
          if (valueSymbols.count(destOp)) {
            if (auto srcValue = valueSymbols.lookup(srcOp))
              valueSymbols.insert(destOp, srcValue);
            else
              valueSymbols.insert(destOp, op.getSrc());

            isNewValue = true;
          }
        })
        // Delete the local variables defined in `if` statements.
        .Case<mlir::scf::IfOp>([&](auto op) {
          // Handle `then` region.
          deleteLocalVar(localVars, users, op.getThenRegion());

          // Handle `else` region.
          deleteLocalVar(localVars, users, op.getElseRegion());
        });

    // Assume the `a` is a local variable, which one user is `b = a`.
    // Replace `a` with its value, then erase this user from `users` container.
    // Although `a = 1` is also the user of `a`, don't replace it.
    if (users.contains(&op) && !isNewValue)
      for (auto operand : op.getOperands())
        if (auto value = valueSymbols.lookup(operand.getDefiningOp())) {
          op.replaceUsesOfWith(operand, value);
          users.erase(&op);
        }
  }
}

std::unique_ptr<mlir::Pass> circt::moore::createDeleteLocalVarPass() {
  return std::make_unique<DeleteLocalVarPass>();
}

void DeleteLocalVarPass::runOnOperation() {
  getOperation()->walk([&](ProcedureOp procedureOp) {
    // Used to collect all users of local variables.
    DenseSet<Operation *> users;

    // Used to collect all local variables.
    DenseSet<Operation *> localVars;

    DLV dlv;
    dlv.deleteLocalVar(localVars, users, procedureOp.getBodyRegion());

    // Erase the redundant users of local varaibles.
    for (auto *user : users)
      user->erase();

    // Erase the local variables.
    for (auto *var : localVars)
      var->erase();

    return WalkResult::advance();
  });
}
