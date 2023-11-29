/*++Copyright (c) 2015 Microsoft Corporation--*/

#include </Users/luisa/z3/src/api/c++/z3++.h>

using namespace z3;

// updates

expr AToB(context& c, expr x, expr go) {
  return x;
}

expr BToA(context& c, expr x, expr go) {
  return c.bv_val(0, 32);
}

expr BToB(context& c, expr x, expr go) {
  return x + 1;
}

// guards, tautology if none, unless updates

expr trans_AB(context& c, expr x, expr go) {
   return go;
}

expr trans_BB(context& c, expr x, expr go) {
    return x!=5;
}

expr trans_BA(context& c, expr x, expr go) {
    return x==5;
}

int main() {

  context c; 

  func_decl IA = c.function("IA", c.bv_sort(32),  c.bool_sort());
  func_decl IB = c.function("IB", c.bv_sort(32), c.bool_sort());

  solver s(c);

  expr x = c.bv_const("x", 32);
  expr go = c.bool_const("go");
  expr go_prime = c.bool_const("go_prime");



  s.add(forall(go, IA(0)));

  s.add(forall(go, x, implies(IA(x), IB(AToB(c, x, go)))));
  s.add(forall(go, x, implies((IB(x) && trans_BA(c, x, go)), IA(BToA(c, x, go)))));
  s.add(forall(go, x, implies((IB(x) && trans_BB(c, x, go)), IB(BToB(c,  x, go)))));
  s.add(forall(go, x, implies((IA(x) && x != 0), 0)));

  std::cout<<s.check()<<std::endl;

  llvm::outs()<<"------------------------ SOLVER ------------------------"<<"\n";
  llvm::outs()<<solver.to_smt2()<<"\n";
  llvm::outs()<<"--------------------------------------------------------"<<"\n";

  return 0;
}