#include "fsm-verification.h"
#include "circt/Dialect/FSM/FSMGraph.h"


using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace std;

void print_cfg(CFG *cfg){
  llvm::outs()<<"------------------------ CFG ------------------------"<<"\n";
  llvm::outs()<<"------------------------ VARIABLES ------------------------"<<"\n";
  for(variable v : cfg->variables){
    llvm::outs()<<"Variable: "<<v.name<<" with init value "<<v.initValue<<"\n";
  }
  llvm::outs()<<"------------------------ INPUTS ------------------------"<<"\n";
  for(inputs i : cfg->inputs){
    llvm::outs()<<"Input: "<<i.name<<"\n";
  }
  llvm::outs()<<"Initial State: "<<cfg->initialState<<"\n";
  llvm::outs()<<"------------------------ OUTPUTS ------------------------"<<"\n";
  for(outputs o : cfg->outputs){
    llvm::outs()<<"Output: "<<o.name<<"\n";
  }
  llvm::outs()<<"------------------------ TRANSITIONS ------------------------"<<"\n";
  for(transition *t : cfg->transitions){
    llvm::outs()<<"Transition from "<<t->from<<" to "<<t->to<<"\n";
    llvm::outs()<<"Guards: ";
    for(string s : *(t->guards)){
      llvm::outs()<<s<<" ";
    }
    llvm::outs()<<"\n";
    llvm::outs()<<"Updates: ";
    for(string s : *(t->var_updates)){
      llvm::outs()<<s<<" ";
    }
    llvm::outs()<<"\n";
  }
}

void manage_block_in_out(Block &op, CFG *cfg){
  // llvm::outs()<<"BLOCKSSSSSSSSSS"<<"\n";
  // llvm::outs() << op <<"\n";
  // auto init = op.getAttr("initialState").cast<StringAttr>().getValue().str().c_str();
  // cfg->initialState = init;
  auto v = op.getArguments();
  auto types = op.getArgumentTypes();

  int idx_arg=0;

  llvm::outs()<< "Arguments:" << v.size() <<"\n";
  for(auto e: v){
    // llvm::outs()<< e.getImpl() <<"\n";
    string s = "arg";// + to_string(e.getLoc());
    inputs i;
    i.name = s;
    cfg->inputs.push_back(i);
    idx_arg++;
    // llvm::outs()<< "this is the argument "<< s <<"\n";
    // inputs in;
    // in.name = e;
    // cfg->inputs.push_back(in);
  }
}

void store_variable_declaration(Operation &op, CFG *cfg){
  variable v;
  v.name = op.getAttr("name").cast<StringAttr>().getValue().str();
  v.initValue = 0;
  cfg->variables.push_back(v);
  // llvm::outs() << op.getResult(0) << endl;;u
  // v.initValue = op.getAttr("initValue").cast<StringAttr>().getValue();
}

string manage_comb_exp(Operation &op){
  llvm::outs()<<"MANAGE COMB EXP\n";
  string log_exp = "";
  if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.add")){
    int n = op.getNumOperands();
    // llvm::outs()<<"Number of operands: "<<n<<"\n";
    auto ops = op.getOperands();
    for(int i = 0; i < n; i++){
      // llvm::outs()<<"\t\t\tOperand: "<<op.getName().getStringRef().str().c_str()<<"\n";
      if(!strcmp(ops[i].getDefiningOp()->getName().getStringRef().str().c_str(), "fsm.variable")){
        log_exp += ops[i].getDefiningOp()->getAttr("name").cast<StringAttr>().getValue().str();
      } else if (auto op = dyn_cast<hw::ConstantOp>(ops[i].getDefiningOp())){
        int in = op.getValue().getZExtValue();
        string s = std::to_string(in);
        log_exp += s;
      }
      if(i!=n-1){
        log_exp += " + ";
      }
    }

  } 
  else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.and")){
  int n = op.getNumOperands();

      llvm::outs()<<"Number of operands: "<<n<<"\n";
      auto ops = op.getOperands();
      for(int i = 0; i < n; i++){
        llvm::outs()<<"\t\t\tOperand: "<<op<<"\n";
        if(ops[i].getDefiningOp()!=NULL){
          if(!strcmp(ops[i].getDefiningOp()->getName().getStringRef().str().c_str(), "fsm.variable")){
            log_exp += ops[i].getDefiningOp()->getAttr("name").cast<StringAttr>().getValue().str();
          } else if (auto op = dyn_cast<hw::ConstantOp>(ops[i].getDefiningOp())){
            int in = op.getValue().getZExtValue();
            string s = std::to_string(in);
            log_exp += s;
          } 
        } else {
          // string s = to_string(ops[i].getImpl());//to_string(ops[i].getImpl());
          // log_exp += s;
        }
        if(i!=n-1){
          log_exp += " and ";
        }
      }
  }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.concat")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.divs")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.divu")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.extract")){

  // }
  else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.icmp")){
    int n = op.getNumOperands();
    // llvm::outs()<<"Number of operands: "<<n<<"\n";
    auto ops = op.getOperands();
    for(auto [i, op1]: llvm::enumerate(ops)){
      // llvm::outs()<<"\t\t\tOperand: "<<op.getName().getStringRef().str().c_str()<<"\n";
      if(auto var_op = dyn_cast<fsm::VariableOp>(op1.getDefiningOp())){
        
        // !strcmp(ops[i].getDefiningOp()->getName().getStringRef().str().c_str(), "fsm.variable")){
        log_exp += var_op.getName();

      } else if (auto op2 = dyn_cast<hw::ConstantOp>(op1.getDefiningOp())){
        int in = op2.getValue().getZExtValue();
        string s = std::to_string(in);
        log_exp += s;
      }
      if(i!=n-1){
        log_exp += " == ";
      }
    }
  }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.mods")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.modu")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.mul")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.mux")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.or")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.parity")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.replicate")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.shl")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.shrs")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.shru")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.sub")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.truth_table")){

  // }
  // else if(!strcmp(op.getName().getStringRef().str().c_str(), "comb.xor")){

  // }

  // llvm::outs()<<"EXP: "<<log_exp<<"\n";

  return log_exp;
  
}

void manage_output_region(Region &region, CFG *cfg){
	// llvm::outs() << "OUTPUT\n";
  //   for (Block &bl : region.getBlocks()){
  //     for(Operation &op : bl.getOperations()){
	// 	if(!strcmp(op.getName().getStringRef().str().c_str(), "fsm.output")){
	// 		llvm::outs() << "\t\tOuptuts "<< op.getNumResults() << "\n";
	// 	} else {
	// 		cout << "MANAGE OUTPUT VAL\n";
	// 	}
  //     }
  // }
}

void manage_guard_region(Region &region, transition *t){
	// llvm::outs() << "GUARD\n";
  for (Block &bl : region.getBlocks()){
      for(Operation &op : bl.getOperations()){
        if (auto op1 = dyn_cast<fsm::ReturnOp>(op)){
          // TODO HERE
          // string s = manage_update_op()
          // t->var_updates->push_back(s);
        } else {
          string s = manage_comb_exp(op);
          t->guards->push_back(s);
        }
      }
  }
}

void manage_action_region(Region &region, transition *t){
	// llvm::outs() << "ACTION\n";
  for (Block &bl : region.getBlocks()){
      for(Operation &op : bl.getOperations()){
        if (auto op1 = dyn_cast<fsm::UpdateOp>(op)){
          // if(auto op2 = dyn_cast<fsm::VariableOp>((dyn_cast<fsm::UpdateOp>(op)).getVariable())){
            // auto op1 = dyn_cast<fsm::UpdateOp>(op).getVariable()
          // string name = (dyn_cast<fsm::VariableOp>(op1)).getName().str();
          // llvm::outs()<<name<<"\n";

          // }
        
          // string s2 = op.getAttr("value").cast<StringAttr>().getValue().str();
          // string s = s1 + " = ";
          // TODO HERE
          // string s = manage_update_op()
          // t->var_updates->push_back(s);
        } else {
          string s = manage_comb_exp(op);
          t->var_updates->push_back(s);
        }
      }
  }
}


void manage_transitions_region(Region &region, string current_state, CFG *cfg){
// llvm::outs() << "TRANSITIONS\n";
    for (Block &bl : region.getBlocks()){
      for(Operation &op : bl.getOperations()){
      if(!strcmp(op.getName().getStringRef().str().c_str(), "fsm.transition")){
        transition *t = new transition();
        t->var_updates = new std::vector< std::string >();
        t->guards = new std::vector< std::string >();
        t->from = current_state;
        t->to = op.getAttr("nextState").cast<FlatSymbolRefAttr>().getValue().str();
        // llvm::outs() << "\t\tTransition from "<< t->from << " to " << t->to << "\n";
        MutableArrayRef<Region> regions = op.getRegions();
        manage_guard_region(regions[0], t);
        manage_action_region(regions[1], t);
        cfg->transitions.push_back(t);
      } else {
        cout << "ERROR WITH TRANSITIONS on op"<< op.getName().getStringRef().str().c_str() << "\n";
      }
      }
  }
}

void manage_state(Operation &op, CFG *cfg){
  // llvm::outs() << "STATE OP ANALYSIS: " << op.getName() << "\n";
  // for (auto arg : op.getOperands()){
  //   llvm::outs() << "With operand: " << arg << "\n";
  // }
  // llvm::outs() << "With name: " << op.getInherentAttr("sym_name") << "\n";
	MutableArrayRef<Region> regions = op.getRegions();
	// llvm::outs() << "REGIONS are "<< regions.size() <<"\n";
  manage_output_region(regions[0], cfg);
  manage_transitions_region(regions[1], op.getAttr("sym_name").cast<StringAttr>().getValue().str(), cfg);
}



void explore_nested_blocks(Operation &op, int level, CFG *cfg){
  for (Region &region : op.getRegions()) {
    for (Block &block : region) {
      manage_block_in_out(block, cfg);
      for (Operation &op : block) {
        llvm::outs()<<op.getName().getStringRef().str().c_str()<<"\n";
        if (!strcmp(op.getName().getStringRef().str().c_str(), "fsm.state")){
          manage_state(op, cfg);
        } else 
        if (!strcmp(op.getName().getStringRef().str().c_str(), "fsm.variable")){
          store_variable_declaration(op, cfg);
        } 
        else if (!strcmp(op.getName().getStringRef().str().c_str(), "fsm.machine")){
          llvm::outs()<< "machine region" <<"\n";
          // manage_machine_in_out(op, cfg); 
        }
        explore_nested_blocks(op, level+1, cfg);
      }
    }
  }
}


void parse_fsm(string input_file){

  DialectRegistry registry;

  CFG *cfg = new CFG();

  // registerAllDialects(registry);

  // clang-format off
  registry.insert<
    comb::CombDialect,
    fsm::FSMDialect,
    hw::HWDialect
  >();

  cout << "parsing:\n" << input_file << endl;

  MLIRContext context(registry);

  // Parse the MLIR code into a module.
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(input_file, &context);


  int it = 0;

  for (Operation &op : module.get()) {
    explore_nested_blocks(op, it, cfg);
  }

  print_cfg(cfg);

}

int main(int argc, char **argv){
  
  InitLLVM y(argc, argv);

  parse_fsm("/Users/luisa/circt/integration_test/Dialect/FSM/variable/top.mlir");

  return 0;

}