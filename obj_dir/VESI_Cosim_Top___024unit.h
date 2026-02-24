// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VESI_Cosim_Top.h for the primary calling header

#ifndef VERILATED_VESI_COSIM_TOP___024UNIT_H_
#define VERILATED_VESI_COSIM_TOP___024UNIT_H_  // guard

#include "verilated.h"


class VESI_Cosim_Top__Syms;

class alignas(VL_CACHE_LINE_BYTES) VESI_Cosim_Top___024unit final : public VerilatedModule {
  public:

    // INTERNAL VARIABLES
    VESI_Cosim_Top__Syms* const vlSymsp;

    // CONSTRUCTORS
    VESI_Cosim_Top___024unit(VESI_Cosim_Top__Syms* symsp, const char* v__name);
    ~VESI_Cosim_Top___024unit();
    VL_UNCOPYABLE(VESI_Cosim_Top___024unit);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
