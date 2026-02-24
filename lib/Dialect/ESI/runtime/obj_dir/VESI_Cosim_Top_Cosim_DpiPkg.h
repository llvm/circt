// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VESI_Cosim_Top.h for the primary calling header

#ifndef VERILATED_VESI_COSIM_TOP_COSIM_DPIPKG_H_
#define VERILATED_VESI_COSIM_TOP_COSIM_DPIPKG_H_  // guard

#include "verilated.h"


class VESI_Cosim_Top__Syms;

class alignas(VL_CACHE_LINE_BYTES) VESI_Cosim_Top_Cosim_DpiPkg final : public VerilatedModule {
  public:

    // INTERNAL VARIABLES
    VESI_Cosim_Top__Syms* const vlSymsp;

    // CONSTRUCTORS
    VESI_Cosim_Top_Cosim_DpiPkg(VESI_Cosim_Top__Syms* symsp, const char* v__name);
    ~VESI_Cosim_Top_Cosim_DpiPkg();
    VL_UNCOPYABLE(VESI_Cosim_Top_Cosim_DpiPkg);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
