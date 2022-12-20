
namespace {
class LatticeValue {
    enum Kind {
        /// A value with a yet-to-be-determined value.  This is an unprocessed value.
        Unknown,

        // Value is live, but derives from an indeterminate value.
        Undefined,

        // Value is live and derived from a controlled value.
        Valid,

        // Value is live and derived from an external signal.
        External
    }

    LatticeValue() : tag(Kind::Unknown) {}
};
}

undef = undef op *
valid = valid op valid
Extern = Extern op (valid, Extern)

undef = reg no-init
undef = reg init sig x, val y (x | y is undef)
extern = reg init sig x, val y (x | y is extern and x & y is not undef)


map<Value, LatticeValue>


for each M : Modules {
    for each e : M {
        visitor(e);
    }
}
while (!WL.empty()) {
    visit(wl.pop());
}


namespace {
    struct UndefAnalysisPass : public UndefAnalysisBase<UndefAnalysisPass> {
        void runOnOperation() override;
        void markBlockExecutable(Block* block);
        void visitOperation(Operation* op);
    };
}

UndefAnlaysisPass::runOnOperation() {
  auto circuit = getOperation();
  LLVM_DEBUG(
      { logger.startLine() << "IMConstProp : " << circuit.getName() << "\n"; });
  // Mark the input ports of the top-level modules as being external.  We ignore all other public modules.
  auto top = circuit.getMainModule();
      for (auto port : top.getBodyBlock()->getArguments())
        markExternal(port);
  markBlockExecutable(module.getBodyBlock());

  // If a value changed lattice state then reprocess any of its users.
  while (!changedLatticeValueWorklist.empty()) {
    Value changedVal = changedLatticeValueWorklist.pop_back_val();
    for (Operation *user : changedVal.getUsers()) {
      if (isBlockExecutable(user->getBlock()))
        visitOperation(user);
    }
  }

}
