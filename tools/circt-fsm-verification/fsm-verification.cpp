#include "fsm-verification.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include </Users/luisa/z3/src/api/z3.h>
#include "llvm/ADT/DenseMap.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace std;

void print_cfg(CFG *cfg){
  llvm::outs()<<"------------------------ CFG ------------------------"<<"\n";
  llvm::outs()<<"------------------------ VARIABLES ------------------------"<<"\n";
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

/// Populate the table of combinational transforms
// void populateCombTransformTable() {
//   // Assign non-ambiguous pairs (that don't require type clarification)
//   this->combTransformTable = {
//       {comb::AddOp::getOperationName(),
//        [](auto op1, auto op2) { return op1 + op2; }},
//       {comb::AndOp::getOperationName(),
//        [](auto op1, auto op2) { return z3::operator&(op1, op2); }},
//       {comb::ConcatOp::getOperationName(),
//        [](auto op1, auto op2) { return z3::concat(op1, op2); }},
//       {comb::DivSOp::getOperationName(),
//        [](auto op1, auto op2) { return z3::operator/(op1, op2); }},
//       {comb::DivUOp::getOperationName(),
//        [](auto op1, auto op2) { return z3::udiv(op1, op2); }},
//       {comb::ModSOp::getOperationName(),
//        [](auto op1, auto op2) { return z3::smod(op1, op2); }},
//       {comb::ModUOp::getOperationName(),
//        [](auto op1, auto op2) { return z3::urem(op1, op2); }},
//       {comb::MulOp::getOperationName(),
//        [](auto op1, auto op2) { return op1 * op2; }},
//       {comb::OrOp::getOperationName(),
//        [](auto op1, auto op2) { return op1 | op2; }},
//       {comb::ShlOp::getOperationName(),
//        [](auto op1, auto op2) { return z3::shl(op1, op2); }},
//       {comb::ShrSOp::getOperationName(),
//        [](auto op1, auto op2) { return z3::ashr(op1, op2); }},
//       {comb::SubOp::getOperationName(),
//        [](auto op1, auto op2) { return op1 - op2; }},
//       {comb::XorOp::getOperationName(),
//        [](auto op1, auto op2) { return op1 ^ op2; }}};

//   // Add ambiguous pairs - the type of these lambdas is ambiguous, so they
//   // require casting
//   this->combTransformTable.insert(std::pair(
//       comb::ExtractOp::getOperationName(),
//       (std::function<z3::expr(const z3::expr &, uint32_t, int)>)[](
//           const z3::expr &op1, uint32_t lowBit, int width) {
//         return op1.extract(lowBit + width - 1, lowBit);
//       }));

//   this->combTransformTable.insert(std::pair(
//       comb::ICmpOp::getOperationName(),
//       (std::function<z3::expr(circt::comb::ICmpPredicate, const z3::expr &,
//                               const z3::expr &)>)[](
//           circt::comb::ICmpPredicate predicate, auto lhsExpr, auto rhsExpr) {
//         // TODO: clean up and cut down on return points, re-add
//         // bvtobool as well
//         switch (predicate) {
//         case circt::comb::ICmpPredicate::eq:
//           return lhsExpr == rhsExpr;
//           break;
//         case circt::comb::ICmpPredicate::ne:
//           return lhsExpr != rhsExpr;
//           break;
//         case circt::comb::ICmpPredicate::slt:
//           return (z3::slt(lhsExpr, rhsExpr));
//           break;
//         case circt::comb::ICmpPredicate::sle:
//           return (z3::sle(lhsExpr, rhsExpr));
//           break;
//         case circt::comb::ICmpPredicate::sgt:
//           return (z3::sgt(lhsExpr, rhsExpr));
//           break;
//         case circt::comb::ICmpPredicate::sge:
//           return (z3::sge(lhsExpr, rhsExpr));
//           break;
//         case circt::comb::ICmpPredicate::ult:
//           return (z3::ult(lhsExpr, rhsExpr));
//           break;
//         case circt::comb::ICmpPredicate::ule:
//           return (z3::ule(lhsExpr, rhsExpr));
//           break;
//         case circt::comb::ICmpPredicate::ugt:
//           return (z3::ugt(lhsExpr, rhsExpr));
//           break;
//         case circt::comb::ICmpPredicate::uge:
//           return (z3::uge(lhsExpr, rhsExpr));
//           break;
//         // Multi-valued logic comparisons are not supported.
//         case circt::comb::ICmpPredicate::ceq:
//         case circt::comb::ICmpPredicate::weq:
//         case circt::comb::ICmpPredicate::cne:
//         case circt::comb::ICmpPredicate::wne:
//           assert(false);
//         };
//       }));
//   this->combTransformTable.insert(std::pair(
//       comb::MuxOp::getOperationName(),
//       (std::function<z3::expr(const z3::expr &, const z3::expr &,
//                               const z3::expr &)>)[this](
//           auto condExpr, auto tvalue, auto fvalue) {
//         return z3::ite(bvToBool(condExpr), tvalue, fvalue);
//       }));
//   this->combTransformTable.insert(
//       std::pair(comb::ParityOp::getOperationName(), [](auto op1) {
//         assert(false && "ParityOp not currently supported");
//         // TODO
//         // unsigned width = inputExpr.get_sort().bv_size();
//         unsigned width = 5;
//         // input has 1 or more bits
//         z3::expr parity = op1.extract(0, 0);
//         // calculate parity with every other bit
//         for (unsigned int i = 1; i < width; i++) {
//           parity = parity ^ op1.extract(i, i);
//         }
//         return parity;
//       }));
//   // Don't allow ExtractOp
//   this->combTransformTable.insert(
//       std::pair(comb::ExtractOp::getOperationName(), [](auto op1) {
//         assert(false && "ExtractOp not supported");
//         return op1;
//       }));
// };



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
          log_exp += (cfg->map.find_as(ops[i]));
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
      for(auto a: block.getArguments()){
        llvm::outs()<<"Argument: "<<a<<"\n";
        cfg->map.insert(std::pair(a, "arg"));
      }
      for (Operation &op : block) {
        // llvm::outs()<<op.getName().getStringRef().str().c_str()<<"\n";
        if (!strcmp(op.getName().getStringRef().str().c_str(), "fsm.state")){
          manage_state(op, cfg);
        } else if(auto const_op = dyn_cast<hw::ConstantOp>(op)){
          llvm::outs()<<"HW num results: "<<op.getNumResults()<<"\n";
          cfg->map.insert(pair<mlir::Value, string>(op.getResult(0), "hw.const"));
        } else if(auto const_op = dyn_cast<fsm::VariableOp>(op)){
          llvm::outs()<<"FSM num results: "<<op.getNumResults()<<"\n";
          cfg->map.insert(pair<mlir::Value, string>(op.getResult(0), op.getAttr("name").cast<StringAttr>().getValue().str()));
        }
        else if (!strcmp(op.getName().getStringRef().str().c_str(), "fsm.machine")){
          llvm::outs()<< "machine region" <<"\n";
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

  llvm::DenseMap<mlir::Value, string> map;

  cfg->map = map;

  cout << "parsing:\n" << input_file << endl;

  MLIRContext context(registry);

  // Parse the MLIR code into a module.
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(input_file, &context);


  int it = 0;

  explore_nested_blocks(module.get()[0], it, cfg);

  print_cfg(cfg);

}

int main(int argc, char **argv){
  
  InitLLVM y(argc, argv);

  parse_fsm("/Users/luisa/circt/integration_test/Dialect/FSM/variable/top.mlir");

  return 0;

}