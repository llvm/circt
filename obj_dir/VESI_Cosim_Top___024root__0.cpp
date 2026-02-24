// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VESI_Cosim_Top.h for the primary calling header

#include "VESI_Cosim_Top__pch.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void VESI_Cosim_Top___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

void VESI_Cosim_Top___024root___eval_triggers__act(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_triggers__act\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered[0U] = (QData)((IData)(
                                                    ((IData)(vlSelfRef.clk) 
                                                     & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__clk__0)))));
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        VESI_Cosim_Top___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
}

bool VESI_Cosim_Top___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___trigger_anySet__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        if (in[n]) {
            return (1U);
        }
        n = ((IData)(1U) + n);
    } while ((1U > n));
    return (0U);
}

void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc12_TOP__Cosim_DpiPkg(std::string endpoint_id, const VlUnpacked<CData/*7:0*/, 1> &data, IData/*31:0*/ data_size, IData/*31:0*/ &cosim_ep_tryput__Vfuncrtn);
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc8_TOP__Cosim_DpiPkg(std::string endpoint_id, const VlUnpacked<CData/*7:0*/, 2> &data, IData/*31:0*/ data_size, IData/*31:0*/ &cosim_ep_tryput__Vfuncrtn);
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc6_TOP__Cosim_DpiPkg(std::string endpoint_id, const VlUnpacked<CData/*7:0*/, 7> &data, IData/*31:0*/ data_size, IData/*31:0*/ &cosim_ep_tryput__Vfuncrtn);
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc2_TOP__Cosim_DpiPkg(std::string endpoint_id, const VlUnpacked<CData/*7:0*/, 16> &data, IData/*31:0*/ data_size, IData/*31:0*/ &cosim_ep_tryput__Vfuncrtn);
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc13_TOP__Cosim_DpiPkg(std::string endpoint_id, const VlUnpacked<CData/*7:0*/, 8> &data, IData/*31:0*/ data_size, IData/*31:0*/ &cosim_ep_tryput__Vfuncrtn);
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc12_TOP__Cosim_DpiPkg(std::string endpoint_id, VlUnpacked<CData/*7:0*/, 1> &data, IData/*31:0*/ &data_size, IData/*31:0*/ &cosim_ep_tryget__Vfuncrtn);
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc13_TOP__Cosim_DpiPkg(std::string endpoint_id, VlUnpacked<CData/*7:0*/, 13> &data, IData/*31:0*/ &data_size, IData/*31:0*/ &cosim_ep_tryget__Vfuncrtn);
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc8_TOP__Cosim_DpiPkg(std::string endpoint_id, VlUnpacked<CData/*7:0*/, 2> &data, IData/*31:0*/ &data_size, IData/*31:0*/ &cosim_ep_tryget__Vfuncrtn);
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc7_TOP__Cosim_DpiPkg(std::string endpoint_id, VlUnpacked<CData/*7:0*/, 3> &data, IData/*31:0*/ &data_size, IData/*31:0*/ &cosim_ep_tryget__Vfuncrtn);
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc6_TOP__Cosim_DpiPkg(std::string endpoint_id, VlUnpacked<CData/*7:0*/, 7> &data, IData/*31:0*/ &data_size, IData/*31:0*/ &cosim_ep_tryget__Vfuncrtn);
void VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc4_TOP__Cosim_DpiPkg(std::string endpoint_id, VlUnpacked<CData/*7:0*/, 9> &data, IData/*31:0*/ &data_size, IData/*31:0*/ &cosim_ep_tryget__Vfuncrtn);
extern const VlWide<10>/*319:0*/ VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0;

void VESI_Cosim_Top___024root___nba_sequent__TOP__0(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___nba_sequent__TOP__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    VlUnpacked<CData/*7:0*/, 1> ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOutBuffer;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOutBuffer[__Vi0] = 0;
    }
    VlUnpacked<CData/*7:0*/, 1> ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOutBuffer;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOutBuffer[__Vi0] = 0;
    }
    VlUnpacked<CData/*7:0*/, 1> ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOutBuffer;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOutBuffer[__Vi0] = 0;
    }
    VlUnpacked<CData/*7:0*/, 1> ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOutBuffer;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOutBuffer[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc13__2__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc13__2__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 13> __Vfunc_cosim_ep_tryget__Vdpioc13__2__data;
    for (int __Vi0 = 0; __Vi0 < 13; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc13__2__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc13__2__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryput__Vdpioc13__5__Vfuncout;
    __Vfunc_cosim_ep_tryput__Vdpioc13__5__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc12__8__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc12__8__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 1> __Vfunc_cosim_ep_tryget__Vdpioc12__8__data;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc12__8__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc12__8__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc12__8__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryput__Vdpioc12__11__Vfuncout;
    __Vfunc_cosim_ep_tryput__Vdpioc12__11__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc11__14__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc11__14__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 1> __Vfunc_cosim_ep_tryget__Vdpioc11__14__data;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc11__14__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc11__14__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc11__14__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryput__Vdpioc11__17__Vfuncout;
    __Vfunc_cosim_ep_tryput__Vdpioc11__17__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc10__20__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc10__20__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 1> __Vfunc_cosim_ep_tryget__Vdpioc10__20__data;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc10__20__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc10__20__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc10__20__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryput__Vdpioc10__23__Vfuncout;
    __Vfunc_cosim_ep_tryput__Vdpioc10__23__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc9__26__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc9__26__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 1> __Vfunc_cosim_ep_tryget__Vdpioc9__26__data;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc9__26__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc9__26__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc9__26__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryput__Vdpioc9__29__Vfuncout;
    __Vfunc_cosim_ep_tryput__Vdpioc9__29__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc8__32__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc8__32__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 2> __Vfunc_cosim_ep_tryget__Vdpioc8__32__data;
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc8__32__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc8__32__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc8__32__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryput__Vdpioc8__35__Vfuncout;
    __Vfunc_cosim_ep_tryput__Vdpioc8__35__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc7__38__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc7__38__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 3> __Vfunc_cosim_ep_tryget__Vdpioc7__38__data;
    for (int __Vi0 = 0; __Vi0 < 3; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc7__38__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc7__38__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc7__38__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryput__Vdpioc7__41__Vfuncout;
    __Vfunc_cosim_ep_tryput__Vdpioc7__41__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc6__44__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc6__44__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 7> __Vfunc_cosim_ep_tryget__Vdpioc6__44__data;
    for (int __Vi0 = 0; __Vi0 < 7; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc6__44__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc6__44__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc6__44__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryput__Vdpioc6__47__Vfuncout;
    __Vfunc_cosim_ep_tryput__Vdpioc6__47__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc5__50__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc5__50__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 1> __Vfunc_cosim_ep_tryget__Vdpioc5__50__data;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc5__50__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc5__50__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc5__50__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryput__Vdpioc5__53__Vfuncout;
    __Vfunc_cosim_ep_tryput__Vdpioc5__53__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc4__56__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc4__56__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 9> __Vfunc_cosim_ep_tryget__Vdpioc4__56__data;
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc4__56__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc4__56__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc4__56__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc3__62__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc3__62__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 1> __Vfunc_cosim_ep_tryget__Vdpioc3__62__data;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc3__62__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc3__62__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc3__62__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc2__68__Vfuncout;
    __Vfunc_cosim_ep_tryget__Vdpioc2__68__Vfuncout = 0;
    VlUnpacked<CData/*7:0*/, 1> __Vfunc_cosim_ep_tryget__Vdpioc2__68__data;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        __Vfunc_cosim_ep_tryget__Vdpioc2__68__data[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_cosim_ep_tryget__Vdpioc2__68__data_size;
    __Vfunc_cosim_ep_tryget__Vdpioc2__68__data_size = 0;
    IData/*31:0*/ __Vfunc_cosim_ep_tryput__Vdpioc2__71__Vfuncout;
    __Vfunc_cosim_ep_tryput__Vdpioc2__71__Vfuncout = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state;
    __Vdly__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg = 0;
    VlWide<16>/*511:0*/ __Vtemp_2;
    // Body
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state 
        = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state;
    __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid;
    __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg;
    __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid;
    if ((1U & (~ (IData)(vlSelfRef.rst)))) {
        if (vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg) {
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc12_TOP__Cosim_DpiPkg("loopback_inst[0].mysvc_send.send"s, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__DataInBuffer, 1U, __Vfunc_cosim_ep_tryput__Vdpioc11__17__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryput__Vdpioc11__17__Vfuncout;
            if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:65: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_mysvc_send_send.unnamedblk2: cosim_ep_tryput(loopback_inst[0].mysvc_send.send, *,           1) = %11d Error! (Data lost)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_send_send__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 65, "");
            }
        }
    }
    if ((1U & (~ (IData)(vlSelfRef.rst)))) {
        if (vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg) {
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc12_TOP__Cosim_DpiPkg("loopback_inst[1].mysvc_send.send"s, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__DataInBuffer, 1U, __Vfunc_cosim_ep_tryput__Vdpioc9__29__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryput__Vdpioc9__29__Vfuncout;
            if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:65: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_mysvc_send_send.unnamedblk2: cosim_ep_tryput(loopback_inst[1].mysvc_send.send, *,           1) = %11d Error! (Data lost)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_send_send__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 65, "");
            }
        }
    }
    if ((1U & (~ (IData)(vlSelfRef.rst)))) {
        if (vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg) {
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc8_TOP__Cosim_DpiPkg("func1.result"s, vlSelfRef.ESI_Cosim_Top__DOT__func1_result__DOT__DataInBuffer, 2U, __Vfunc_cosim_ep_tryput__Vdpioc8__35__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryput__Vdpioc8__35__Vfuncout;
            if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:65: Assertion failed in %NESI_Cosim_Top.func1_result.unnamedblk2: cosim_ep_tryput(func1.result, *,           2) = %11d Error! (Data lost)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__func1_result__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 65, "");
            }
        }
    }
    if ((1U & (~ (IData)(vlSelfRef.rst)))) {
        if (vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg) {
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc12_TOP__Cosim_DpiPkg("loopback_inst[0].loopback_fromhw.send"s, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__DataInBuffer, 1U, __Vfunc_cosim_ep_tryput__Vdpioc12__11__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryput__Vdpioc12__11__Vfuncout;
            if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:65: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_loopback_fromhw_send.unnamedblk2: cosim_ep_tryput(loopback_inst[0].loopback_fromhw.send, *,           1) = %11d Error! (Data lost)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 65, "");
            }
        }
    }
    if ((1U & (~ (IData)(vlSelfRef.rst)))) {
        if (vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg) {
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc12_TOP__Cosim_DpiPkg("loopback_inst[1].loopback_fromhw.send"s, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__DataInBuffer, 1U, __Vfunc_cosim_ep_tryput__Vdpioc10__23__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryput__Vdpioc10__23__Vfuncout;
            if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:65: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_loopback_fromhw_send.unnamedblk2: cosim_ep_tryput(loopback_inst[1].loopback_fromhw.send, *,           1) = %11d Error! (Data lost)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 65, "");
            }
        }
    }
    if ((1U & (~ (IData)(vlSelfRef.rst)))) {
        if (vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg) {
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc8_TOP__Cosim_DpiPkg("structFunc.result"s, vlSelfRef.ESI_Cosim_Top__DOT__structFunc_result__DOT__DataInBuffer, 2U, __Vfunc_cosim_ep_tryput__Vdpioc7__41__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryput__Vdpioc7__41__Vfuncout;
            if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:65: Assertion failed in %NESI_Cosim_Top.structFunc_result.unnamedblk2: cosim_ep_tryput(structFunc.result, *,           2) = %11d Error! (Data lost)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__structFunc_result__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 65, "");
            }
        }
    }
    if ((1U & (~ (IData)(vlSelfRef.rst)))) {
        if (vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg) {
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc8_TOP__Cosim_DpiPkg("arrayFunc.result"s, vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_result__DOT__DataInBuffer, 2U, __Vfunc_cosim_ep_tryput__Vdpioc5__53__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryput__Vdpioc5__53__Vfuncout;
            if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:65: Assertion failed in %NESI_Cosim_Top.arrayFunc_result.unnamedblk2: cosim_ep_tryput(arrayFunc.result, *,           2) = %11d Error! (Data lost)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_result__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 65, "");
            }
        }
    }
    if ((1U & (~ (IData)(vlSelfRef.rst)))) {
        if (vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg) {
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc6_TOP__Cosim_DpiPkg("oddStructFunc.result"s, vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer, 7U, __Vfunc_cosim_ep_tryput__Vdpioc6__47__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryput__Vdpioc6__47__Vfuncout;
            if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:65: Assertion failed in %NESI_Cosim_Top.oddStructFunc_result.unnamedblk2: cosim_ep_tryput(oddStructFunc.result, *,           7) = %11d Error! (Data lost)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 65, "");
            }
        }
    }
    if ((1U & (~ (IData)(vlSelfRef.rst)))) {
        if (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg) {
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc2_TOP__Cosim_DpiPkg("__cosim_cycle_count.result"s, vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer, 0x00000010U, __Vfunc_cosim_ep_tryput__Vdpioc2__71__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryput__Vdpioc2__71__Vfuncout;
            if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:65: Assertion failed in %NESI_Cosim_Top.__cycle_counter.resp_ep.unnamedblk2: cosim_ep_tryput(__cosim_cycle_count.result, *,          16) = %11d Error! (Data lost)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 65, "");
            }
        }
    }
    if ((1U & (~ (IData)(vlSelfRef.rst)))) {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__data_valid_0) 
             | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state))) {
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryput__Vdpioc13_TOP__Cosim_DpiPkg("__cosim_mmio_read_write.result"s, vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer, 8U, __Vfunc_cosim_ep_tryput__Vdpioc13__5__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryput__Vdpioc13__5__Vfuncout;
            if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:65: Assertion failed in %NESI_Cosim_Top.__cosim_mmio_read_write_result.unnamedblk2: cosim_ep_tryput(__cosim_mmio_read_write.result, *,           8) = %11d Error! (Data lost)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 65, "");
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit = 1U;
            __Vfunc_cosim_ep_tryget__Vdpioc12__8__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc12__8__data[0U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc12_TOP__Cosim_DpiPkg("loopback_inst[0].loopback_tohw.recv"s, __Vfunc_cosim_ep_tryget__Vdpioc12__8__data, __Vfunc_cosim_ep_tryget__Vdpioc12__8__data_size, __Vfunc_cosim_ep_tryget__Vdpioc12__8__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc12__8__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc12__8__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc12__8__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_loopback_tohw_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[0].loopback_tohw.recv, *,           1 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_loopback_tohw_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[0].loopback_tohw.recv, *,           1 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc)) {
                if ((1U == vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_loopback_tohw_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[0].loopback_tohw.recv, *,           1 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit = 1U;
            __Vfunc_cosim_ep_tryget__Vdpioc11__14__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc11__14__data[0U] 
                = ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc12_TOP__Cosim_DpiPkg("loopback_inst[0].mysvc_recv.recv"s, __Vfunc_cosim_ep_tryget__Vdpioc11__14__data, __Vfunc_cosim_ep_tryget__Vdpioc11__14__data_size, __Vfunc_cosim_ep_tryget__Vdpioc11__14__Vfuncout);
            ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc11__14__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc11__14__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc11__14__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_mysvc_recv_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[0].mysvc_recv.recv, *,           1 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_mysvc_recv_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[0].mysvc_recv.recv, *,           1 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc)) {
                if ((1U == vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_mysvc_recv_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[0].mysvc_recv.recv, *,           1 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit = 1U;
            __Vfunc_cosim_ep_tryget__Vdpioc10__20__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc10__20__data[0U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc12_TOP__Cosim_DpiPkg("loopback_inst[1].loopback_tohw.recv"s, __Vfunc_cosim_ep_tryget__Vdpioc10__20__data, __Vfunc_cosim_ep_tryget__Vdpioc10__20__data_size, __Vfunc_cosim_ep_tryget__Vdpioc10__20__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc10__20__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc10__20__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc10__20__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_loopback_tohw_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[1].loopback_tohw.recv, *,           1 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_loopback_tohw_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[1].loopback_tohw.recv, *,           1 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__rc)) {
                if ((1U == vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_loopback_tohw_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[1].loopback_tohw.recv, *,           1 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit = 1U;
            __Vfunc_cosim_ep_tryget__Vdpioc9__26__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc9__26__data[0U] 
                = ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc12_TOP__Cosim_DpiPkg("loopback_inst[1].mysvc_recv.recv"s, __Vfunc_cosim_ep_tryget__Vdpioc9__26__data, __Vfunc_cosim_ep_tryget__Vdpioc9__26__data_size, __Vfunc_cosim_ep_tryget__Vdpioc9__26__Vfuncout);
            ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc9__26__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc9__26__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc9__26__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_mysvc_recv_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[1].mysvc_recv.recv, *,           1 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_mysvc_recv_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[1].mysvc_recv.recv, *,           1 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__rc)) {
                if ((1U == vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_mysvc_recv_recv.unnamedblk2: cosim_ep_tryget(loopback_inst[1].mysvc_recv.recv, *,           1 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__data_limit = 1U;
            __Vfunc_cosim_ep_tryget__Vdpioc5__50__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc5__50__data[0U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc12_TOP__Cosim_DpiPkg("arrayFunc.arg"s, __Vfunc_cosim_ep_tryget__Vdpioc5__50__data, __Vfunc_cosim_ep_tryget__Vdpioc5__50__data_size, __Vfunc_cosim_ep_tryget__Vdpioc5__50__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc5__50__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc5__50__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc5__50__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.arrayFunc_arg.unnamedblk2: cosim_ep_tryget(arrayFunc.arg, *,           1 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.arrayFunc_arg.unnamedblk2: cosim_ep_tryget(arrayFunc.arg, *,           1 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__rc)) {
                if ((1U == vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.arrayFunc_arg.unnamedblk2: cosim_ep_tryget(arrayFunc.arg, *,           1 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__data_limit = 1U;
            __Vfunc_cosim_ep_tryget__Vdpioc3__62__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc3__62__data[0U] 
                = ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc12_TOP__Cosim_DpiPkg("__cosim_hostmem_write.result"s, __Vfunc_cosim_ep_tryget__Vdpioc3__62__data, __Vfunc_cosim_ep_tryget__Vdpioc3__62__data_size, __Vfunc_cosim_ep_tryget__Vdpioc3__62__Vfuncout);
            ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc3__62__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc3__62__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc3__62__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_write_result.unnamedblk2: cosim_ep_tryget(__cosim_hostmem_write.result, *,           1 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_write_result.unnamedblk2: cosim_ep_tryget(__cosim_hostmem_write.result, *,           1 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__rc)) {
                if ((1U == vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_write_result.unnamedblk2: cosim_ep_tryget(__cosim_hostmem_write.result, *,           1 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__data_limit = 1U;
            __Vfunc_cosim_ep_tryget__Vdpioc2__68__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc2__68__data[0U] 
                = ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc12_TOP__Cosim_DpiPkg("__cosim_cycle_count.arg"s, __Vfunc_cosim_ep_tryget__Vdpioc2__68__data, __Vfunc_cosim_ep_tryget__Vdpioc2__68__data_size, __Vfunc_cosim_ep_tryget__Vdpioc2__68__Vfuncout);
            ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc2__68__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc2__68__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc2__68__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.__cycle_counter.req_ep.unnamedblk2: cosim_ep_tryget(__cosim_cycle_count.arg, *,           1 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.__cycle_counter.req_ep.unnamedblk2: cosim_ep_tryget(__cosim_cycle_count.arg, *,           1 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__rc)) {
                if ((1U == vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.__cycle_counter.req_ep.unnamedblk2: cosim_ep_tryget(__cosim_cycle_count.arg, *,           1 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__data_limit = 0x0000000dU;
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[0x0000000cU] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [0x0000000cU];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[0x0000000bU] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [0x0000000bU];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[0x0000000aU] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [0x0000000aU];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[9U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [9U];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[8U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [8U];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[7U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [7U];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[6U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [6U];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[5U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [5U];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[4U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [4U];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[3U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [3U];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[2U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [2U];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[1U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [1U];
            __Vfunc_cosim_ep_tryget__Vdpioc13__2__data[0U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc13_TOP__Cosim_DpiPkg("__cosim_mmio_read_write.arg"s, __Vfunc_cosim_ep_tryget__Vdpioc13__2__data, __Vfunc_cosim_ep_tryget__Vdpioc13__2__data_size, __Vfunc_cosim_ep_tryget__Vdpioc13__2__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[0x0000000cU] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [0x0000000cU];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[0x0000000bU] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [0x0000000bU];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[0x0000000aU] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [0x0000000aU];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[9U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [9U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[8U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [8U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[7U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [7U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[6U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [6U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[5U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [5U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[4U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [4U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[3U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [3U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[2U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [2U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[1U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [1U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc13__2__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.__cosim_mmio_read_write_arg.unnamedblk2: cosim_ep_tryget(__cosim_mmio_read_write.arg, *,          13 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.__cosim_mmio_read_write_arg.unnamedblk2: cosim_ep_tryget(__cosim_mmio_read_write.arg, *,          13 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__rc)) {
                if ((0x0000000dU == vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.__cosim_mmio_read_write_arg.unnamedblk2: cosim_ep_tryget(__cosim_mmio_read_write.arg, *,          13 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__data_limit = 2U;
            __Vfunc_cosim_ep_tryget__Vdpioc8__32__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc8__32__data[1U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOutBuffer
                [1U];
            __Vfunc_cosim_ep_tryget__Vdpioc8__32__data[0U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc8_TOP__Cosim_DpiPkg("func1.arg"s, __Vfunc_cosim_ep_tryget__Vdpioc8__32__data, __Vfunc_cosim_ep_tryget__Vdpioc8__32__data_size, __Vfunc_cosim_ep_tryget__Vdpioc8__32__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOutBuffer[1U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc8__32__data
                [1U];
            vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc8__32__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc8__32__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc8__32__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.func1_arg.unnamedblk2: cosim_ep_tryget(func1.arg, *,           2 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.func1_arg.unnamedblk2: cosim_ep_tryget(func1.arg, *,           2 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__rc)) {
                if ((2U == vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.func1_arg.unnamedblk2: cosim_ep_tryget(func1.arg, *,           2 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__data_limit = 3U;
            __Vfunc_cosim_ep_tryget__Vdpioc7__38__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc7__38__data[2U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer
                [2U];
            __Vfunc_cosim_ep_tryget__Vdpioc7__38__data[1U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer
                [1U];
            __Vfunc_cosim_ep_tryget__Vdpioc7__38__data[0U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc7_TOP__Cosim_DpiPkg("structFunc.arg"s, __Vfunc_cosim_ep_tryget__Vdpioc7__38__data, __Vfunc_cosim_ep_tryget__Vdpioc7__38__data_size, __Vfunc_cosim_ep_tryget__Vdpioc7__38__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer[2U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc7__38__data
                [2U];
            vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer[1U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc7__38__data
                [1U];
            vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc7__38__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc7__38__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc7__38__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.structFunc_arg.unnamedblk2: cosim_ep_tryget(structFunc.arg, *,           3 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.structFunc_arg.unnamedblk2: cosim_ep_tryget(structFunc.arg, *,           3 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__rc)) {
                if ((3U == vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.structFunc_arg.unnamedblk2: cosim_ep_tryget(structFunc.arg, *,           3 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__data_limit = 7U;
            __Vfunc_cosim_ep_tryget__Vdpioc6__44__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc6__44__data[6U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                [6U];
            __Vfunc_cosim_ep_tryget__Vdpioc6__44__data[5U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                [5U];
            __Vfunc_cosim_ep_tryget__Vdpioc6__44__data[4U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                [4U];
            __Vfunc_cosim_ep_tryget__Vdpioc6__44__data[3U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                [3U];
            __Vfunc_cosim_ep_tryget__Vdpioc6__44__data[2U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                [2U];
            __Vfunc_cosim_ep_tryget__Vdpioc6__44__data[1U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                [1U];
            __Vfunc_cosim_ep_tryget__Vdpioc6__44__data[0U] 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc6_TOP__Cosim_DpiPkg("oddStructFunc.arg"s, __Vfunc_cosim_ep_tryget__Vdpioc6__44__data, __Vfunc_cosim_ep_tryget__Vdpioc6__44__data_size, __Vfunc_cosim_ep_tryget__Vdpioc6__44__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer[6U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc6__44__data
                [6U];
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer[5U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc6__44__data
                [5U];
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer[4U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc6__44__data
                [4U];
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer[3U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc6__44__data
                [3U];
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer[2U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc6__44__data
                [2U];
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer[1U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc6__44__data
                [1U];
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc6__44__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc6__44__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc6__44__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.oddStructFunc_arg.unnamedblk2: cosim_ep_tryget(oddStructFunc.arg, *,           7 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.oddStructFunc_arg.unnamedblk2: cosim_ep_tryget(oddStructFunc.arg, *,           7 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__rc)) {
                if ((7U == vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.oddStructFunc_arg.unnamedblk2: cosim_ep_tryget(oddStructFunc.arg, *,           7 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_ready))) {
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid = 0U;
        }
        if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid)) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_ready))) {
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__data_limit = 9U;
            __Vfunc_cosim_ep_tryget__Vdpioc4__56__data_size 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__data_limit;
            __Vfunc_cosim_ep_tryget__Vdpioc4__56__data[8U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                [8U];
            __Vfunc_cosim_ep_tryget__Vdpioc4__56__data[7U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                [7U];
            __Vfunc_cosim_ep_tryget__Vdpioc4__56__data[6U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                [6U];
            __Vfunc_cosim_ep_tryget__Vdpioc4__56__data[5U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                [5U];
            __Vfunc_cosim_ep_tryget__Vdpioc4__56__data[4U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                [4U];
            __Vfunc_cosim_ep_tryget__Vdpioc4__56__data[3U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                [3U];
            __Vfunc_cosim_ep_tryget__Vdpioc4__56__data[2U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                [2U];
            __Vfunc_cosim_ep_tryget__Vdpioc4__56__data[1U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                [1U];
            __Vfunc_cosim_ep_tryget__Vdpioc4__56__data[0U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                [0U];
            VESI_Cosim_Top_Cosim_DpiPkg____Vdpiimwrap_cosim_ep_tryget__Vdpioc4_TOP__Cosim_DpiPkg("__cosim_hostmem_read_resp.data"s, __Vfunc_cosim_ep_tryget__Vdpioc4__56__data, __Vfunc_cosim_ep_tryget__Vdpioc4__56__data_size, __Vfunc_cosim_ep_tryget__Vdpioc4__56__Vfuncout);
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer[8U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc4__56__data
                [8U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer[7U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc4__56__data
                [7U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer[6U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc4__56__data
                [6U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer[5U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc4__56__data
                [5U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer[4U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc4__56__data
                [4U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer[3U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc4__56__data
                [3U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer[2U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc4__56__data
                [2U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer[1U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc4__56__data
                [1U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer[0U] 
                = __Vfunc_cosim_ep_tryget__Vdpioc4__56__data
                [0U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__data_limit 
                = __Vfunc_cosim_ep_tryget__Vdpioc4__56__data_size;
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__rc 
                = __Vfunc_cosim_ep_tryget__Vdpioc4__56__Vfuncout;
            if (VL_UNLIKELY((VL_GTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:154: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_read_resp_data.unnamedblk2: cosim_ep_tryget(__cosim_hostmem_read_resp.data, *,           9 -> %11d) returned an error (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 154, "");
            } else if (VL_UNLIKELY((VL_LTS_III(32, 0U, vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__rc)))) {
                VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:157: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_read_resp_data.unnamedblk2: cosim_ep_tryget(__cosim_hostmem_read_resp.data, *,           9 -> %11d) had data left over! (%11d)\n",0,
                             64,VL_TIME_UNITED_Q(1),
                             -12,vlSymsp->name(),32,
                             vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__data_limit,
                             32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__rc);
                VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 157, "");
            } else if ((0U == vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__rc)) {
                if ((9U == vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__data_limit)) {
                    __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid = 1U;
                } else if (VL_UNLIKELY(((0U != vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__data_limit)))) {
                    VL_WRITEF_NX("[%0t] %%Error: Cosim_Endpoint.sv:165: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_read_resp_data.unnamedblk2: cosim_ep_tryget(__cosim_hostmem_read_resp.data, *,           9 -> %11d) did not load entire buffer!\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name(),
                                 32,vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__unnamedblk2__DOT__data_limit);
                    VL_STOP_MT("/workspace/circt/build/default/tools/circt/lib/Dialect/ESI/runtime/python/esiaccel/cosim/Cosim_Endpoint.sv", 165, "");
                }
            }
        }
    }
    __Vdly__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state 
        = ((~ (IData)(vlSelfRef.rst)) & ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_1) 
                                         | ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_1)) 
                                            & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state))));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state 
        = ((~ (IData)(vlSelfRef.rst)) & ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_0) 
                                         | ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_0)) 
                                            & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state))));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_ready_reg 
        = (1U & (~ (IData)(vlSelfRef.rst)));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_ready_reg 
        = (1U & (~ (IData)(vlSelfRef.rst)));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_ready_reg 
        = (1U & (~ (IData)(vlSelfRef.rst)));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_ready_reg 
        = (1U & (~ (IData)(vlSelfRef.rst)));
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_ready_reg 
        = (1U & (~ (IData)(vlSelfRef.rst)));
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_ready_reg 
        = (1U & (~ (IData)(vlSelfRef.rst)));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_ready_reg 
        = (1U & (~ (IData)(vlSelfRef.rst)));
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_ready_reg 
        = (1U & (~ (IData)(vlSelfRef.rst)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_ready_reg 
        = (1U & (~ (IData)(vlSelfRef.rst)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_ready_reg 
        = (1U & (~ (IData)(vlSelfRef.rst)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_ready_reg 
        = (1U & (~ (IData)(vlSelfRef.rst)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_ready_reg 
        = ((~ (IData)(vlSelfRef.rst)) & (IData)(vlSelfRef.ESI_Cosim_Top__DOT___ESI_Cosim_UserTopWrapper_mmio_cmd_ready));
    if (vlSelfRef.rst) {
        vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count = 0ULL;
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HeaderMMIO__DOT__cycle_counter__DOT___GEN = 0ULL;
    } else {
        vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
            = (1ULL + vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count);
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HeaderMMIO__DOT__cycle_counter__DOT___GEN 
            = (1ULL + vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HeaderMMIO__DOT__cycle_counter__DOT___GEN);
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid)))) {
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid))) {
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_write_result.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg) {
                __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid)))) {
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid))) {
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_mysvc_recv_recv.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg) {
                __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid)))) {
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid))) {
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_mysvc_recv_recv.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg) {
                __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid)))) {
            __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid))) {
            __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.__cycle_counter.req_ep.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg) {
                __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid)))) {
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid))) {
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.__cosim_hostmem_read_resp_data.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg) {
                __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid)))) {
            __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid))) {
            __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.func1_arg.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg) {
                __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid)))) {
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOutBuffer
                [0U];
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid))) {
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l;
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv;
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOutBuffer
                [0U];
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.loopback_inst5B05D_loopback_tohw_recv.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg) {
                vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l 
                    = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOutBuffer
                    [0U];
                __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg 
                    = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOutBuffer
                    [0U];
                __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid)))) {
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOutBuffer
                [0U];
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid))) {
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l;
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv;
            vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l 
                = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOutBuffer
                [0U];
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.loopback_inst5B15D_loopback_tohw_recv.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg) {
                vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l 
                    = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOutBuffer
                    [0U];
                __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg 
                    = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOutBuffer
                    [0U];
                __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid)))) {
            vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOutBuffer
                [0U];
            __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid))) {
            vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l;
            __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__a_rcv;
            vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l 
                = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOutBuffer
                [0U];
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.arrayFunc_arg.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg) {
                vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l 
                    = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOutBuffer
                    [0U];
                __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_reg 
                    = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOutBuffer
                    [0U];
                __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid)))) {
            vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a;
            __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid))) {
            vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l;
            __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__a_rcv;
            vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l 
                = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a;
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.structFunc_arg.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg) {
                vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l 
                    = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a;
                __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_reg 
                    = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a;
                __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid)))) {
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[0U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[0U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[1U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[1U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[2U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[2U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[3U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[3U];
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid))) {
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[0U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[0U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[1U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[1U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[2U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[2U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[3U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[3U];
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__a_rcv;
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[0U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[0U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[1U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[1U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[2U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[2U];
            vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[3U] 
                = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[3U];
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.__cosim_mmio_read_write_arg.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg) {
                vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[0U] 
                    = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[0U];
                vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[1U] 
                    = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[1U];
                vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[2U] 
                    = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[2U];
                vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l[3U] 
                    = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[3U];
                __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[0U] 
                    = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[0U];
                vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[1U] 
                    = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[1U];
                vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[2U] 
                    = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[2U];
                vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[3U] 
                    = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[3U];
                __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSelfRef.rst) {
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[0U] = 0U;
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[1U] = 0U;
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] = 0U;
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[3U] = 0U;
    } else if (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_0) {
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[0U] 
            = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[0U];
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[1U] 
            = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[1U];
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
            = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[2U];
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[3U] 
            = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[3U];
    }
    if (vlSelfRef.rst) {
        __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid = 0U;
        __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__xmit) 
             & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid)))) {
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a;
            __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__xmit) 
             & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid))) {
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l;
            __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 1U;
            __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__a_rcv;
            vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l 
                = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a;
        } else if (((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__xmit)) 
                    & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.oddStructFunc_arg.out_pipe: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg) {
                vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l 
                    = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a;
                __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid = 1U;
            } else {
                vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                    = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a;
                __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg = 1U;
            }
        }
    }
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_1 
        = ((0x019fU >= (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_0))
            ? vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN
           [vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_0]
            : 0ULL);
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a[0U] 
        = (((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
             [3U] << 0x00000018U) | (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                                     [2U] << 0x00000010U)) 
           | ((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
               [1U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
              [0U]));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a[1U] 
        = (((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
             [7U] << 0x00000018U) | (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
                                     [6U] << 0x00000010U)) 
           | ((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
               [5U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
              [4U]));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a[2U] 
        = (0x000000ffU & vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOutBuffer
           [8U]);
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a 
        = ((vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOutBuffer
            [1U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOutBuffer
           [0U]);
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a 
        = ((vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer
            [2U] << 0x00000010U) | ((vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer
                                     [1U] << 8U) | 
                                    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOutBuffer
                                    [0U]));
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[0U] 
        = (((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
             [3U] << 0x00000018U) | (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                                     [2U] << 0x00000010U)) 
           | ((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
               [1U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
              [0U]));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[1U] 
        = (((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
             [7U] << 0x00000018U) | (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                                     [6U] << 0x00000010U)) 
           | ((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
               [5U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
              [4U]));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[2U] 
        = (((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
             [0x0bU] << 0x00000018U) | (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
                                        [0x0aU] << 0x00000010U)) 
           | ((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
               [9U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
              [8U]));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a[3U] 
        = (1U & vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOutBuffer
           [0x0cU]);
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a 
        = (((QData)((IData)(((0x00070000U & (vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                             [6U] << 0x00000010U)) 
                             | ((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                 [5U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                [4U])))) << 0x00000020U) 
           | (QData)((IData)((((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                [3U] << 0x00000018U) 
                               | (vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                  [2U] << 0x00000010U)) 
                              | ((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                  [1U] << 8U) | vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOutBuffer
                                 [0U])))));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg 
        = __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid 
        = __Vdly__ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__data_valid_0 
        = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__data_valid;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[8U] 
        = (0x000000ffU & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[9U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 8U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000aU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000010U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000bU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000018U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000cU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000020U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000dU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000028U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000eU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000030U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__resp_ep__DOT__DataInBuffer[0x0000000fU] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__cycle_count 
                                  >> 0x00000038U)));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_0 
        = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_fromhw_send__DOT__DataInBuffer[0U] 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_fromhw_send__DOT__DataInBuffer[0U] 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_result__DOT__DataInBuffer[0U] 
        = vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_result__DOT__DataInBuffer[1U] 
        = (0x000000ffU & ((IData)(1U) + (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__x_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_result__DOT__DataInBuffer[0U] 
        = (0x000000ffU & vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_reg);
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_result__DOT__DataInBuffer[1U] 
        = (0x000000ffU & ((IData)(1U) + vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__x_reg));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.__VdfgRegularize_he50b618e_0_7 = (IData)(
                                                       (0U 
                                                        != 
                                                        (0xfffffc00U 
                                                         & vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[2U])));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__xmit 
        = vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_valid_reg;
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__l_valid)) 
                 | (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_ready_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[0U] 
        = (0x000000ffU & ((IData)(1U) + (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg)));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[1U] 
        = (0x000000ffU & ((IData)(2U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                 >> 8U))));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[2U] 
        = (0x000000ffU & ((IData)(2U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                 >> 0x00000010U))));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[3U] 
        = (0x000000ffU & ((IData)(5U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                 >> 0x00000018U))));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[4U] 
        = ((0x00000080U & (((IData)(1U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                   >> 0x00000027U))) 
                           << 7U)) | (0x0000007fU & 
                                      ((IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                >> 0x00000020U)) 
                                       - (IData)(3U))));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[5U] 
        = (0x000000ffU & (((IData)(1U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                  >> 0x00000027U))) 
                          >> 1U));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer[6U] 
        = ((0xf8U & vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_result__DOT__DataInBuffer
            [6U]) | (7U & (((IData)(1U) + (IData)((vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__x_reg 
                                                   >> 0x00000027U))) 
                           >> 9U)));
    __Vtemp_2[0U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[0U];
    __Vtemp_2[1U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[1U];
    __Vtemp_2[2U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[2U];
    __Vtemp_2[3U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[3U];
    __Vtemp_2[4U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[4U];
    __Vtemp_2[5U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[5U];
    __Vtemp_2[6U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[6U];
    __Vtemp_2[7U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[7U];
    __Vtemp_2[8U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[8U];
    __Vtemp_2[9U] = VESI_Cosim_Top__ConstPool__CONST_h2f74ed10_0[9U];
    __Vtemp_2[0x0000000aU] = (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HeaderMMIO__DOT__cycle_counter__DOT___GEN);
    __Vtemp_2[0x0000000bU] = (IData)((vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HeaderMMIO__DOT__cycle_counter__DOT___GEN 
                                      >> 0x00000020U));
    __Vtemp_2[0x0000000cU] = 0U;
    __Vtemp_2[0x0000000dU] = 0U;
    __Vtemp_2[0x0000000eU] = 0U;
    __Vtemp_2[0x0000000fU] = 0U;
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state)
            ? (((QData)((IData)(__Vtemp_2[(((IData)(0x0000003fU) 
                                            + (0x000001ffU 
                                               & VL_SHIFTL_III(9,9,32, 
                                                               (7U 
                                                                & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                                   >> 3U)), 6U))) 
                                           >> 5U)])) 
                << ((0U == (0x0000001fU & VL_SHIFTL_III(9,9,32, 
                                                        (7U 
                                                         & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                            >> 3U)), 6U)))
                     ? 0x00000020U : ((IData)(0x00000040U) 
                                      - (0x0000001fU 
                                         & VL_SHIFTL_III(9,9,32, 
                                                         (7U 
                                                          & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                             >> 3U)), 6U))))) 
               | (((0U == (0x0000001fU & VL_SHIFTL_III(9,9,32, 
                                                       (7U 
                                                        & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                           >> 3U)), 6U)))
                    ? 0ULL : ((QData)((IData)(__Vtemp_2[
                                              (((IData)(0x0000001fU) 
                                                + (0x000001ffU 
                                                   & VL_SHIFTL_III(9,9,32, 
                                                                   (7U 
                                                                    & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                                       >> 3U)), 6U))) 
                                               >> 5U)])) 
                              << ((IData)(0x00000020U) 
                                  - (0x0000001fU & 
                                     VL_SHIFTL_III(9,9,32, 
                                                   (7U 
                                                    & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                       >> 3U)), 6U))))) 
                  | ((QData)((IData)(__Vtemp_2[(0x0000000fU 
                                                & (VL_SHIFTL_III(9,9,32, 
                                                                 (7U 
                                                                  & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                                     >> 3U)), 6U) 
                                                   >> 5U))])) 
                     >> (0x0000001fU & VL_SHIFTL_III(9,9,32, 
                                                     (7U 
                                                      & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_msg_reg[2U] 
                                                         >> 3U)), 6U)))))
            : vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_1);
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__ESI_Manifest_ROM__DOT___GEN_0 
        = (0x000001ffU & (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg[2U] 
                          >> 3U));
    if (vlSelfRef.rst) {
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg[0U] = 0U;
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg[1U] = 0U;
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg[2U] = 0U;
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg[3U] = 0U;
    } else if (vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_1) {
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg[0U] 
            = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[0U];
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg[1U] 
            = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[1U];
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg[2U] 
            = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[2U];
        vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_msg_reg[3U] 
            = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[3U];
    }
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_write_result__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_mysvc_recv_recv__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_mysvc_recv_recv__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcycle_counter__DOT__req_ep__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_hostmem_read_resp_data__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__func1_arg__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B05D_loopback_tohw_recv__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__loopback_inst5B15D_loopback_tohw_recv__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__arrayFunc_arg__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__structFunc_arg__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__out_pipe__DOT__a_rcv 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__oddStructFunc_arg__DOT__DataOut_a_valid));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[0U] 
        = (0x000000ffU & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[1U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 8U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[2U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000010U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[3U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000018U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[4U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000020U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[5U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000028U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[6U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000030U)));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataInBuffer[7U] 
        = (0x000000ffU & (IData)((vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_result__DOT__DataIn 
                                  >> 0x00000038U)));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ESI_Manifest_ROM_Wrapper__DOT__data_valid 
        = vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state 
        = __Vdly__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state;
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[0U] 
        = (IData)((((QData)((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[1U])) 
                    << 0x00000020U) | (QData)((IData)(
                                                      vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[0U]))));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[1U] 
        = (IData)(((((QData)((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[1U])) 
                     << 0x00000020U) | (QData)((IData)(
                                                       vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[0U]))) 
                   >> 0x00000020U));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[2U] 
        = ((0U != (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[2U] 
                   >> 0x0000000aU)) ? (vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[2U] 
                                       - (IData)(0x00000400U))
            : (0x000003ffU & vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[2U]));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT___GEN_1[3U] 
        = (1U & vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_reg[3U]);
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__consume_1 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state) 
           & (~ (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state)));
    vlSelfRef.ESI_Cosim_Top__DOT___ESI_Cosim_UserTopWrapper_mmio_cmd_ready 
        = (1U & (~ (((~ (IData)(vlSelfRef.__VdfgRegularize_he50b618e_0_7)) 
                     & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out0_valid_reg__DOT__state)) 
                    | ((IData)(vlSelfRef.__VdfgRegularize_he50b618e_0_7) 
                       & (IData)(vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__out1_valid_reg__DOT__state)))));
    vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT___ESI_Cosim_UserTopWrapper_mmio_cmd_ready) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__x_valid_reg));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_0 
        = ((~ (IData)(vlSelfRef.__VdfgRegularize_he50b618e_0_7)) 
           & (IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit));
    vlSelfRef.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l0_i0__DOT__will_write_1 
        = ((IData)(vlSelfRef.ESI_Cosim_Top__DOT_____05Fcosim_mmio_read_write_arg__DOT__out_pipe__DOT__xmit) 
           & (IData)(vlSelfRef.__VdfgRegularize_he50b618e_0_7));
}

void VESI_Cosim_Top___024root___eval_nba(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_nba\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VnbaTriggered[0U])) {
        VESI_Cosim_Top___024root___nba_sequent__TOP__0(vlSelf);
    }
}

void VESI_Cosim_Top___024root___trigger_orInto__act(VlUnpacked<QData/*63:0*/, 1> &out, const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___trigger_orInto__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = (out[n] | in[n]);
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool VESI_Cosim_Top___024root___eval_phase__act(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_phase__act\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    VESI_Cosim_Top___024root___eval_triggers__act(vlSelf);
    VESI_Cosim_Top___024root___trigger_orInto__act(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    return (0U);
}

void VESI_Cosim_Top___024root___trigger_clear__act(VlUnpacked<QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___trigger_clear__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool VESI_Cosim_Top___024root___eval_phase__nba(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_phase__nba\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = VESI_Cosim_Top___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        VESI_Cosim_Top___024root___eval_nba(vlSelf);
        VESI_Cosim_Top___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

void VESI_Cosim_Top___024root___eval(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VnbaIterCount;
    // Body
    __VnbaIterCount = 0U;
    do {
        if (VL_UNLIKELY(((0x00000064U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            VESI_Cosim_Top___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
            VL_FATAL_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESI_Cosim_Top.sv", 18, "", "NBA region did not converge after 100 tries");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        vlSelfRef.__VactIterCount = 0U;
        do {
            if (VL_UNLIKELY(((0x00000064U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                VESI_Cosim_Top___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
                VL_FATAL_MT("/tmp/esi-pytest-run-8f8cykc2/hw/ESI_Cosim_Top.sv", 18, "", "Active region did not converge after 100 tries");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
        } while (VESI_Cosim_Top___024root___eval_phase__act(vlSelf));
    } while (VESI_Cosim_Top___024root___eval_phase__nba(vlSelf));
}

#ifdef VL_DEBUG
void VESI_Cosim_Top___024root___eval_debug_assertions(VESI_Cosim_Top___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VESI_Cosim_Top___024root___eval_debug_assertions\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if (VL_UNLIKELY(((vlSelfRef.clk & 0xfeU)))) {
        Verilated::overWidthError("clk");
    }
    if (VL_UNLIKELY(((vlSelfRef.rst & 0xfeU)))) {
        Verilated::overWidthError("rst");
    }
}
#endif  // VL_DEBUG
