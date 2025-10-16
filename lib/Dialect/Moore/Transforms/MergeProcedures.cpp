#include <string>

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Support/LLVM.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "merge-procedures"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_MERGEPROCEDURES
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;
using mlir::IRRewriter;

namespace {

struct PrintMatchFailure : mlir::RewriterBase::Listener {
  void
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) override {
    Diagnostic diag(loc, mlir::DiagnosticSeverity::Remark);
    reasonCallback(diag);
  }
};

struct MergeProceduresPass
    : public circt::moore::impl::MergeProceduresBase<MergeProceduresPass> {
  void runOnOperation() override;
  LogicalResult TryMerge(IRRewriter &rewriter,
                         ArrayRef<ProcedureOp> procedures);
};

} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createMergeProceduresPass() {
  return std::make_unique<MergeProceduresPass>();
}

LogicalResult MergeProceduresPass::TryMerge(IRRewriter &rewriter,
                                            ArrayRef<ProcedureOp> procedures) {
  if (procedures.size() <= 1)
    return success();

  // Only merge always/always_ff procedures if their wait_events are identical.
  SmallVector<WaitEventOp> wait_events;
  wait_events.reserve(procedures.size());
  for (ProcedureOp proc : procedures) {
    auto proc_wait_events = proc.getOps<WaitEventOp>();
    if (!hasSingleElement(proc_wait_events)) {
      return rewriter.notifyMatchFailure(proc, "expected a single wait event");
    }
    wait_events.push_back(*proc_wait_events.begin());
  }

  // We only merge procedures with a single block only. Otherwise it needs more
  // work to maintain correct control flow.
  for (ProcedureOp proc : procedures) {
    if (proc.getBody().getBlocks().size() != 1) {
      return rewriter.notifyMatchFailure(
          proc, "can only merge procedures with a single block");
    }
  }

  // Compare wait event ops by string serialization.
  WaitEventOp first_wait_event = wait_events[0];
  auto is_equivalent = [&](Value lhs, Value rhs) -> LogicalResult {
    // lhs refers to 'first_wait_event'.
    auto producer = lhs.getDefiningOp();
    // If it's a block argument, check for value equivalence.
    if (!producer)
      return success(lhs == rhs);
    // If the value was produced within the wait event, it's equivalent.
    if (first_wait_event->isProperAncestor(producer))
      return success();
    // If the value was produced outside of the wait event, check for value
    // equivalence.
    return success(lhs == rhs);
  };

  // Check if wait_events are equivalent.
  for (WaitEventOp wait_event : wait_events) {
    if (!mlir::OperationEquivalence::isEquivalentTo(
            first_wait_event, wait_event,
            /*checkEquivalent=*/is_equivalent,
            /*markEquivalent=*/nullptr,
            mlir::OperationEquivalence::Flags::IgnoreLocations)) {
      return rewriter.notifyMatchFailure(wait_event, "wait event mismatch");
    }
  }

  // Remove all but the first wait_event.
  for (WaitEventOp wait_event : llvm::ArrayRef(wait_events).drop_front()) {
    wait_event.erase();
  }

  // Merge all procedures into the first one.
  ProcedureOp first_proc = procedures[0];
  Block *dst_proc_block = &first_proc.getBody().getBlocks().front();
  for (ProcedureOp proc : llvm::ArrayRef(procedures).drop_front()) {
    Block *src_proc_block = &proc.getBody().getBlocks().front();
    src_proc_block->getTerminator()->erase();
    rewriter.inlineBlockBefore(src_proc_block, dst_proc_block,
                               // insert before block terminator.
                               std::prev(dst_proc_block->end()));
    // Delete all but the first procedure.
    proc.erase();
  }

  return success();
}

void MergeProceduresPass::runOnOperation() {
  /*
  Sample rewrite

  moore.module @bug(...) {
    moore.procedure always_ff {
      moore.wait_event {
        %3 = moore.read %wr_clk_0 : <l1>
        moore.detect_event posedge %3 : l1
      }
      ... // A
    }
    moore.procedure always_ff {
      moore.wait_event {
        %3 = moore.read %wr_clk_0 : <l1>
        moore.detect_event posedge %3 : l1
      }
      ... // B
    }

  ~>

  moore.module @bug(...) {
    moore.procedure always_ff {
      moore.wait_event {
        %3 = moore.read %wr_clk_0 : <l1>
        moore.detect_event posedge %3 : l1
      }
      ... // A
      ... // B
    }
    ...
  */

  // Collect all moore.procedures and group by their kind. We try to merge
  // procedures of each kind individually, so we merge multiple always_ff
  // procedures, but we don't merge always_ff and always together.
  llvm::DenseMap<moore::ProcedureKind, SmallVector<ProcedureOp>> procedures;
  getOperation().walk(
      [&](ProcedureOp proc) { procedures[proc.getKind()].push_back(proc); });

  IRRewriter rewriter(&getContext());
  PrintMatchFailure listener;
  LLVM_DEBUG(rewriter.setListener(&listener););

  // Try to merge procedures of each kind. Failing to merge is not an error -
  // instead the error may later surface when lowering arc to llvm.
  for (moore::ProcedureKind kind :
       {moore::ProcedureKind::AlwaysFF, moore::ProcedureKind::Always}) {
    if (failed(TryMerge(rewriter, procedures[kind]))) {
      getOperation().emitWarning("could not merge procedures of kind ")
          << static_cast<uint32_t>(kind);
    }
  }
}
