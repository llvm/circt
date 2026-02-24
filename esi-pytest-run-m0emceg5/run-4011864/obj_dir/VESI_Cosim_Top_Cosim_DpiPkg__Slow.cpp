// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VESI_Cosim_Top.h for the primary calling header

#include "VESI_Cosim_Top__pch.h"

void VESI_Cosim_Top_Cosim_DpiPkg___ctor_var_reset(VESI_Cosim_Top_Cosim_DpiPkg* vlSelf);

VESI_Cosim_Top_Cosim_DpiPkg::VESI_Cosim_Top_Cosim_DpiPkg(VESI_Cosim_Top__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    VESI_Cosim_Top_Cosim_DpiPkg___ctor_var_reset(this);
}

void VESI_Cosim_Top_Cosim_DpiPkg::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

VESI_Cosim_Top_Cosim_DpiPkg::~VESI_Cosim_Top_Cosim_DpiPkg() {
}
