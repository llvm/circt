#include "fsm-verification.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include </Users/luisa/z3/src/api/c++/z3++.h>
#include "llvm/ADT/DenseMap.h"

#define VERBOSE 1

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace std;
using namespace z3;

void print_cfg(CFG *cfg){
  llvm::outs()<<"------------------------ CFG ------------------------"<<"\n";
  llvm::outs()<<"------------------------ VARIABLES ------------------------"<<"\n";
  llvm::outs()<<"------------------------ INPUTS ------------------------"<<"\n";
  for(auto i : cfg->inputs){
    llvm::outs()<<"Input: "<<cfg->map.at(i)<<"\n";
  }
  llvm::outs()<<"Initial State: "<<cfg->initialState<<"\n";
  llvm::outs()<<"------------------------ OUTPUTS ------------------------"<<"\n";
  for(outputs o : cfg->outputs){
    llvm::outs()<<"Output: "<<o.name<<" from "<<o.state_from<<"\n";
  }
  llvm::outs()<<"------------------------ TRANSITIONS ------------------------"<<"\n";
  for(transition *t : cfg->transitions){
    llvm::outs()<<"Transition from "<<t->from<<" to "<<t->to<<"\n";
    llvm::outs()<<"Guards: ";
    for(string s : *(t->guards)){
      llvm::outs()<<s<<"\n";
    }
    llvm::outs()<<"\n";
    llvm::outs()<<"Updates: ";
    for(string s : *(t->var_updates)){
      llvm::outs()<<s<<"\n";
    }
    llvm::outs()<<"\n";
  }
}

void printSolverAssertions(z3::solver& solver) {
  llvm::outs()<<"------------------------ SOLVER ------------------------"<<"\n";
    model m = solver.get_model();
    std::cout << m << "\n";
    // traversing the model
    for (unsigned i = 0; i < m.size(); i++) {
        func_decl v = m[i];
        // this problem contains only constants
        assert(v.arity() == 0); 
        std::cout << v.name() << " = " << m.get_const_interp(v) << "\n";
    }
}


// Populate the table of combinational transforms
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


string manage_comb_exp(Operation &op, context& c, CFG *cfg, solver& solver, int idx){
  string s;
  llvm::outs()<<"comb expression "<< op.getName().getStringRef().str().c_str() <<"\n";
  string log_exp = "";
  if(auto add = dyn_cast<comb::AddOp>(op)){
    int n = add.getNumOperands();
    auto ops = add.getOperands();
    mlir::Value new_var = add.getResult();
    cfg->map.insert({new_var, "res"+to_string(idx)});
    s = s + "res"+to_string(idx) + " = ";
    for(int i = 0; i < n; i++){
      llvm::outs()<<"\tAddOp argument "<<cfg->map.at(ops[i])<<"\n";
      s = s + cfg->map.at(ops[i]);
      if(i < n-1){
        s = s + " + ";
      }
      
      // llvm::outs()<<"\t\t\tOperand: "<<op.getName().getStringRef().str().c_str()<<"\n";
    }
  } 
  else if(auto and_op = dyn_cast<comb::AndOp>(op)){
    int n = and_op.getNumOperands();
    auto ops = and_op.getOperands();
    mlir::Value new_var = and_op.getResult();
    cfg->map.insert({new_var, "res"+to_string(idx)});
    s = s + "res"+to_string(idx) + " = ";
    for(int i = 0; i < n; i++){
      llvm::outs()<<"\tAndOp argument "<<cfg->map.at(ops[i])<<"\n";
      // llvm::outs()<<"\t\t\tOperand: "<<op.getName().getStringRef().str().c_str()<<"\n";
      s = s + cfg->map.at(ops[i]);
      if(i < n-1){
        s = s + " & ";
      }
    }
  }
  else if(auto icmp = dyn_cast<comb::ICmpOp>(op)){
    int n = icmp.getNumOperands();
    auto ops = icmp.getOperands();
    mlir::Value new_var = icmp.getResult();
    cfg->map.insert({new_var, "res"+to_string(idx)});
    s = s + "res"+to_string(idx) + " = ";
    for(int i = 0; i < n; i++){
      llvm::outs()<<"\tICmpOp argument "<< icmp.getPredicate()<<" " <<cfg->map.at(ops[i])<<"\n";
      // llvm::outs()<<"\t\t\tOperand: "<<op.getName().getStringRef().str().c_str()<<"\n";
      s = s + cfg->map.at(ops[i]);
      if(i < n-1){
        s = s + " == ";
      }
      mlir::Value new_var = icmp.getResult();
      cfg->map.insert({new_var, "res"+to_string(idx)});
    }
  }
  
  return s;
  
}

void manage_output_region(Region &region, CFG *cfg, context& ctx, solver& solver, string current_state){
    for (Block &bl : region.getBlocks()){
      for(Operation &op : bl.getOperations()){
        if(auto output_op = dyn_cast<fsm::OutputOp>(op)){
          auto ops = output_op.getOperands();
          for (auto o : ops){
            outputs out;
            out.state_from = current_state;
            out.name = o;
            cfg->outputs.push_back(out);
          }
        }
      }
  }
}

int manage_guard_region(Region &region, transition *t, context& c, CFG *cfg, solver& solver){
	// llvm::outs() << "GUARD\n";
  int num_op = 0;
  for (Block &bl : region.getBlocks()){
      for(Operation &op : bl.getOperations()){
        if (auto op1 = dyn_cast<fsm::ReturnOp>(op)){
          mlir::Value res = op1.getOperand();
          t->guards->push_back(cfg->map.at(res));
        } else {
          string s = manage_comb_exp(op, c, cfg, solver, num_op);
          t->guards->push_back(s);
          num_op++;
        }
      }
  }
  return num_op;
}

int manage_action_region(Region &region, transition *t, context& c, CFG *cfg, solver& solver){
	// llvm::outs() << "ACTION\n";
  int num_action = 0;
  for (Block &bl : region.getBlocks()){
      for(Operation &op : bl.getOperations()){
        if (auto op1 = dyn_cast<fsm::UpdateOp>(op)){  
          string last_update = cfg->map.at(op1.getVariable()) + " + " + cfg->map.at(op1.getValue());
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
          string s = manage_comb_exp(op, c, cfg, solver, num_action);
          t->var_updates->push_back(s);
          num_action++;
        }
      }
  }
  return num_action;
}


void manage_transitions_region(Region &region, func_decl& inv_state, CFG *cfg, context& c, solver& solver, string current_state){
// llvm::outs() << "TRANSITIONS\n";
  expr x = c.bv_const("x", 32);
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
        int guards = manage_guard_region(regions[0], t, c, cfg, solver);
        int actions = manage_action_region(regions[1], t, c, cfg, solver);

        cfg->transitions.push_back(t);
        
      } else {
        cout << "ERROR WITH TRANSITIONS on op"<< op.getName().getStringRef().str().c_str() << "\n";
      }
      }
  }
}

void manage_state(Operation &op, CFG *cfg, context& ctx, solver& solver){
  func_decl I = ctx.function(op.getAttr("sym_name").cast<StringAttr>().getValue().str().c_str(), ctx.bv_sort(32), ctx.bool_sort());
	MutableArrayRef<Region> regions = op.getRegions();
  solver.add(I(0));
  manage_output_region(regions[0], cfg, ctx, solver, op.getAttr("sym_name").cast<StringAttr>().getValue().str());
  manage_transitions_region(regions[1], I, cfg, ctx, solver, op.getAttr("sym_name").cast<StringAttr>().getValue().str());
}



void explore_nested_blocks(Operation &op, int num_args, CFG *cfg, context& ctx, solver& solver){
  for (Region &region : op.getRegions()) {
    for (Block &block : region) {
      for(auto a: block.getArguments()){
        cfg->inputs.push_back(a);
        cfg->map.insert({a, "arg"+to_string(num_args)});
        num_args++;
      }
      for (Operation &op : block) {
        if (!strcmp(op.getName().getStringRef().str().c_str(), "fsm.state")){
          manage_state(op, cfg, ctx, solver);
        } else if(auto const_op = dyn_cast<hw::ConstantOp>(op)){
          cfg->map.insert({const_op.getResult(), "hw.const"+to_string(num_args)});
          num_args++;
        } else if(auto var_op = dyn_cast<fsm::VariableOp>(op)){
          cfg->map.insert({var_op.getResult(), var_op.getName().str()});
        } else if(auto machine = dyn_cast<fsm::MachineOp>(op)){
          cfg->initialState = machine.getInitialState();
        }
        explore_nested_blocks(op, num_args, cfg, ctx, solver);
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
  OwningOpRef<ModuleOp> module = mlir::parseSourceFile<ModuleOp>(input_file, &context);


  int it = 0;

  z3::context ctx;

  solver s(ctx);


  explore_nested_blocks(module.get()[0], it, cfg, ctx, s);

  print_cfg(cfg);

  // printSolverAssertions(s);

}

int main(int argc, char **argv){
  
  InitLLVM y(argc, argv);

  parse_fsm("/Users/luisa/circt/integration_test/Dialect/FSM/variable/top.mlir");

  return 0;

}




