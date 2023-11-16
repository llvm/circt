/*++Copyright (c) 2015 Microsoft Corporation--*/

#include </Users/luisa/z3/src/api/c++/z3++.h>

using namespace z3;

// updates

expr AToB(context& c, expr x) {
  return x;
}

expr BToA(context& c, expr x) {
  return c.bv_val(0, 32);
}

expr BToB(context& c, expr x) {
  return x + 1;
}

// guards, tautology if none, unless updates

expr trans_AB(context& c, expr x) {
   return x || !x;
}

expr trans_BB(context& c, expr x) {
    return x!=5;
}

expr trans_BA(context& c, expr x) {
    return x==5;
}

int main() {

    context c; 
    z3::sort cn = c.int_sort();
    z3::sort in = c.bool_sort();

    func_decl IA = c.function("IA", c.bv_sort(32), c.bool_sort());
    func_decl IB = c.function("IB", c.bv_sort(32), c.bool_sort());


    solver s(c);

    expr x = c.bv_const("x", 32);

    s.add(IA(0));

    s.add(forall(x, implies(IA(x), IB(AToB(c, x)))));
    s.add(forall(x, implies((IB(x) && trans_BA(c, x)), IA(BToA(c, x)))));
    s.add(forall(x, implies((IB(x) && trans_BB(c, x)), IB(BToB(c,  x)))));
    s.add(forall(x, implies((IA(x) && x != 0), 0)));
  
    std::cout<<s.check()<<std::endl;

    return 0;
}