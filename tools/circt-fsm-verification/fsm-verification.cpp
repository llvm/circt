#include "fsm-verification.h"
#include "circt/Dialect/FSM/FSMGraph.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace std;

struct CFG{
  std::vector<int> states;
  std::vector<std::vector<std::string> > var_updates;
  std::vector<std::vector<std::string> > guards;
  std::vector<int> trans_true;
  std::vector<int> trans_false;
};

void manage_output_region(Region &region){
	llvm::outs() << "OUTPUT\n";
    for (Block &bl : region.getBlocks()){
      for(Operation &op : bl.getOperations()){
		if(!strcmp(op.getName().getStringRef().str().c_str(), "fsm.output")){
			llvm::outs() << "\t\tOuptuts "<< op.getNumResults() << "\n";
		} else {
			cout << "MANAGE OUTPUT VAL\n";
		}
      }
  }
}

void manage_guard_region(Region &region){
	llvm::outs() << "GUARD\n";
}

void manage_action_region(Region &region){
	llvm::outs() << "ACTION\n";
}


void manage_transitions_region(Region &region){
llvm::outs() << "TRANSITIONS\n";
    for (Block &bl : region.getBlocks()){
      for(Operation &op : bl.getOperations()){
		if(!strcmp(op.getName().getStringRef().str().c_str(), "fsm.transition")){
			llvm::outs() << "TRANSITION OP ANALYSIS: " << op.getName() << "\n";
			llvm::outs() << "With next state: " << op.getInherentAttr("nextState") << "\n";
				MutableArrayRef<Region> regions = op.getRegions();
				llvm::outs() << "REGIONS are "<< regions.size() <<"\n";
			manage_guard_region(regions[0]);
			manage_action_region(regions[1]);
		} else {
			cout << "ERROR WITH TRANSITIONS on op"<< op.getName().getStringRef().str().c_str() << "\n";
		}
      }
  }
}

void manage_state(Operation &op){
  llvm::outs() << "STATE OP ANALYSIS: " << op.getName() << "\n";
  for (auto arg : op.getOperands()){
    llvm::outs() << "With operand: " << arg << "\n";
  }
  llvm::outs() << "With name: " << op.getInherentAttr("sym_name") << "\n";
	MutableArrayRef<Region> regions = op.getRegions();
	llvm::outs() << "REGIONS are "<< regions.size() <<"\n";
  manage_output_region(regions[0]);
  manage_transitions_region(regions[1]);
}

// string convert_comb_exp(Operation &op){
//   llvm::outs() << "Operation Inside: " << op.getName() << "\n";
//   for(auto attr : op.getAttrs()){
//     llvm::outs() << "Attributes: " << attr << "\n";
//   }
// }


void explore_region(Region &region){

//     for (Block &bl : region.getBlocks()){
//       for(Operation &op : bl.getOperations()){
//         llvm::outs() << "\t\t\tOperation Inside: " << op.getName() << "\n";
//         // if (!strcmp(op.getName().getStringRef().str().c_str(), "comb.and"))
//         //   convert_comb_exp(op);
//       }
//   }
}

void explore_nested_blocks(Operation &op, int level){
  for (Region &region : op.getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (!strcmp(op.getName().getStringRef().str().c_str(), "fsm.state")){
          manage_state(op);
        } else if (!strcmp(op.getName().getStringRef().str().c_str(), "fsm.transition")){
        //   llvm::outs() << "\tTransition: " << op.getName() << ", level: " <<level<< "\n";\
        //   for (auto arg : op.getOperands()){
        //     llvm::outs() << "\tWith operand: " << arg << "\n";
        //   }
        //   llvm::outs() << "\tWith next state: " << op.getInherentAttr("nextState") << "\n";

        //   llvm::outs() << "\tWith number of regions: " << op.getNumRegions() << "\n";
        //   MutableArrayRef<Region> regions = op.getRegions();
        //   llvm::outs() << "GUARD REGIONS" <<"\n";
        //     explore_region(regions[0]);

        //   llvm::outs() << "ACTION REGIONS" <<"\n";
        //     explore_region(regions[1]);

        }
        explore_nested_blocks(op, level+1);
      }
    }
  }
}


void parse_fsm(string input_file, CFG *cgf){

  DialectRegistry registry;

  // registerAllDialects(registry);

  // clang-format off
  registry.insert<
    comb::CombDialect,
    fsm::FSMDialect,
    hw::HWDialect
  >();

  cout << "parsing:\n" << input_file << endl;

  MLIRContext context(registry);

  // context.getOrLoadDialect<fsm::FSMDialect>();
  // registerBuiltinDialectTranslation(registry);
  // registerLLVMDialectTranslation(registry);
  // registerAllDialects(registry);
  // registerAllPasses();
  // context.appendDialectRegistry(registry);

  string fsm_mlir_code;

  // Parse the MLIR code into a module.
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>("/Users/luisa/circt/integration_test/Dialect/FSM/simple/top.mlir", &context);

  int it = 0;

  for (Operation &op : module.get()) {
    explore_nested_blocks(op, it);
    it++;

  //   // Iterate through the blocks within each region.
  //   llvm::outs() << "Operation: " << op.getName() << "\n";
  //   for (Region &region : op.getRegions()) {
  //     for (Block &block : region) {
  //       for (Operation &op : block) {
  //         llvm::outs() << "Operation Inside: " << op.getName() << "\n";
          
  //       }
  //     }
  //   }
  }

}

int main(int argc, char **argv){
  
  InitLLVM y(argc, argv);

  CFG *cfg = new CFG();

  parse_fsm("/Users/luisa/circt/integration_test/Dialect/FSM/simple/top.mlir", cfg);

  return 0;

}