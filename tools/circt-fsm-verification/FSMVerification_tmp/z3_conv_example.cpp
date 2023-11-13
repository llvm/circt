
/*++Copyright (c) 2015 Microsoft Corporation--*/

#include <z3++.h>

using namespace z3;

// updates

expr AtoB(context* c, expr x) {
  return x;
}

expr BtoA(context* c, expr x) {
  return c->bv_val(0, 32);
}

expr BtoB(context *c, expr x) {
  return x + 1;
}

// guards, tautology if none, unless updates

expr trans_AB(context* c, expr x) {
   return x || !x;
}

expr trans_BB(context* c, expr x) {
    return x!=5;
}

expr trans_BA(context* c, expr x) {
    return x==5;
}

int main() {

    context c; 
    sort cn = c.int_sort();
    sort in = c.bool_sort();

    func_decl IA = c.function("IA", c.bv_sort(32), c.bool_sort());
    func_decl IB = c.function("IB", c.bv_sort(32), c.bool_sort());


    solver s(c);

    expr x = c.bv_const("x", 32);

    s.add(IA(0));

    s.add(forall(x, implies(IA(x), IB(AtoB(c, x)))));
    s.add(forall(x, implies(And(IB(x), transitionBToA(c, x)), IA(BToA(c, x)))));
    s.add(forall(x, implies(And(IB(x), transitionBToB(c, x)), IB(BToB(c,  x)))));
    s.add(forall(x, implies(And(IA(x), x != 0), 0)));
    s.add(forall(x, implies(And(IB(x), x > 5), False)));
  

    std::cout<<solver.check()std::<<endl;

    
    
    return 0;
}

