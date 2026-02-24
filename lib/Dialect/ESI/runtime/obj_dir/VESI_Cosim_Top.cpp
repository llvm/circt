// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "VESI_Cosim_Top__pch.h"

//============================================================
// Constructors

VESI_Cosim_Top::VESI_Cosim_Top(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new VESI_Cosim_Top__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , rst{vlSymsp->TOP.rst}
    , __PVT__Cosim_DpiPkg{vlSymsp->TOP.__PVT__Cosim_DpiPkg}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_0{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_0}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_2{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_2}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i0{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i0}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i1{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i1}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i2{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i2}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i3{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i3}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i4{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i4}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i6{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i6}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i10{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i10}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i11{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i11}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i12{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i12}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i13{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i13}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i14{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i14}
    , __PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i15{vlSymsp->TOP.__PVT__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i15}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

VESI_Cosim_Top::VESI_Cosim_Top(const char* _vcname__)
    : VESI_Cosim_Top(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

VESI_Cosim_Top::~VESI_Cosim_Top() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void VESI_Cosim_Top___024root___eval_debug_assertions(VESI_Cosim_Top___024root* vlSelf);
#endif  // VL_DEBUG
void VESI_Cosim_Top___024root___eval_static(VESI_Cosim_Top___024root* vlSelf);
void VESI_Cosim_Top___024root___eval_initial(VESI_Cosim_Top___024root* vlSelf);
void VESI_Cosim_Top___024root___eval_settle(VESI_Cosim_Top___024root* vlSelf);
void VESI_Cosim_Top___024root___eval(VESI_Cosim_Top___024root* vlSelf);

void VESI_Cosim_Top::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VESI_Cosim_Top::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    VESI_Cosim_Top___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        VESI_Cosim_Top___024root___eval_static(&(vlSymsp->TOP));
        VESI_Cosim_Top___024root___eval_initial(&(vlSymsp->TOP));
        VESI_Cosim_Top___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    VESI_Cosim_Top___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool VESI_Cosim_Top::eventsPending() { return false; }

uint64_t VESI_Cosim_Top::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* VESI_Cosim_Top::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void VESI_Cosim_Top___024root___eval_final(VESI_Cosim_Top___024root* vlSelf);

VL_ATTR_COLD void VESI_Cosim_Top::final() {
    VESI_Cosim_Top___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* VESI_Cosim_Top::hierName() const { return vlSymsp->name(); }
const char* VESI_Cosim_Top::modelName() const { return "VESI_Cosim_Top"; }
unsigned VESI_Cosim_Top::threads() const { return 1; }
void VESI_Cosim_Top::prepareClone() const { contextp()->prepareClone(); }
void VESI_Cosim_Top::atClone() const {
    contextp()->threadPoolpOnClone();
}
