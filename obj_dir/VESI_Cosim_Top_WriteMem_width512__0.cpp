// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VESI_Cosim_Top.h for the primary calling header

#include "VESI_Cosim_Top__pch.h"

void VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512__0(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    QData/*63:0*/ __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN = 0;
    QData/*63:0*/ __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN = 0;
    QData/*63:0*/ __Vdly__address_command__DOT__Counter__DOT___GEN;
    __Vdly__address_command__DOT__Counter__DOT___GEN = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 0;
    // Body
    __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    __Vdly__address_command__DOT__Counter__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid;
    __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i10__DOT__out1_valid_reg__DOT__state)) 
           & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_19_0__DOT__output_channel_ready));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__Counter__DOT___GEN = 0U;
        __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN = 0ULL;
    } else {
        vlSelfRef.__PVT__Counter__DOT___GEN = ((IData)(1U) 
                                               + vlSelfRef.__PVT__Counter__DOT___GEN);
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN));
        }
    }
    vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state 
        = ((~ (IData)(vlSymsp->TOP.rst)) & ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go) 
                                            | ((~ (IData)(vlSelfRef.__PVT__address_command__DOT___GEN_1)) 
                                               & (IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state))));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdCycles_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdIssued_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdResponses_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready));
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__Counter__DOT___GEN = 0ULL;
        __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN = 0ULL;
    } else {
        if (((IData)(vlSelfRef.address_command__DOT____Vcellinp__Counter__increment) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__Counter__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN));
        }
        if (((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__EmitEveryNImpl_4__DOT__ControlReg__DOT__state) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN));
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_512.address_command.pipelineStage_0: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_512.address_command.pipelineStage_2: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_512.address_command.pipelineStage_4: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__address_command__DOT__start_addr = 0ULL;
        vlSelfRef.__PVT__address_command__DOT__flits_total = 0ULL;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
             & (0x00000010U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_msg_reg[2U]))) {
            vlSelfRef.__PVT__address_command__DOT__start_addr 
                = (((QData)((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_msg_reg[1U])) 
                    << 0x00000020U) | (QData)((IData)(
                                                      vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_msg_reg[0U])));
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
             & (0x00000018U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_msg_reg[2U]))) {
            vlSelfRef.__PVT__address_command__DOT__flits_total 
                = (((QData)((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_msg_reg[1U])) 
                    << 0x00000020U) | (QData)((IData)(
                                                      vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_msg_reg[0U])));
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_512.address_command.pipelineStage_3: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_512.address_command.pipelineStage_1: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
            __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l;
            __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_512.address_command.pipelineStage: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
                __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
                __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1 = 0ULL;
    } else if (vlSelfRef.__PVT__address_command__DOT___GEN_1) {
        vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1 
            = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    }
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
        = __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
        = __Vdly__address_command__DOT__Counter__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg)));
    vlSelfRef.address_command__DOT____Vcellinp__Counter__increment 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__TaggedWriteGearboxImpl_4__DOT__ControlReg__DOT__state)) 
           & ((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
              & (vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
                 < vlSelfRef.__PVT__address_command__DOT__flits_total)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready) 
           & ((~ (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                  >> 3U)) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010110__DOT__inp_valid)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
           & ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010110__DOT__inp_valid) 
              & (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                 >> 3U)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready) 
           & ((~ (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                  >> 3U)) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010111__DOT__inp_valid)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
        = __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT___GEN_1 = 
        (vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
         == vlSelfRef.__PVT__address_command__DOT__flits_total);
}

void VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512__1(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__address_command__DOT___GEN = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_valid_reg__DOT__state) 
                                                   & ((IData)(vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready) 
                                                      & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_msg_reg[3U]));
    vlSelfRef.addrCmdResponses_data_ready = ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg)) 
                                             & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_44__DOT__output_channel_ready));
    vlSelfRef.addrCmdIssued_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_44__DOT__output_channel_ready));
    vlSelfRef.addrCmdCycles_data_ready = ((~ (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512.__PVT__pipelineStage__DOT__x_valid_reg)) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_38__DOT__output_channel_ready));
    vlSelfRef.__PVT__address_command__DOT__command_go 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
           & (0x00000020U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_msg_reg[2U]));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdResponses_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdIssued_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdCycles_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg));
}

void VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0__0(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    QData/*63:0*/ __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN = 0;
    QData/*63:0*/ __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN = 0;
    QData/*63:0*/ __Vdly__address_command__DOT__Counter__DOT___GEN;
    __Vdly__address_command__DOT__Counter__DOT___GEN = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 0;
    // Body
    __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    __Vdly__address_command__DOT__Counter__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid;
    __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready 
        = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_valid_reg__DOT__state) 
           & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_27_0__DOT__output_channel_ready));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__Counter__DOT___GEN = 0U;
        __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN = 0ULL;
    } else {
        vlSelfRef.__PVT__Counter__DOT___GEN = ((IData)(1U) 
                                               + vlSelfRef.__PVT__Counter__DOT___GEN);
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN));
        }
    }
    vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state 
        = ((~ (IData)(vlSymsp->TOP.rst)) & ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go) 
                                            | ((~ (IData)(vlSelfRef.__PVT__address_command__DOT___GEN_1)) 
                                               & (IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state))));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdCycles_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdIssued_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdResponses_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready));
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__Counter__DOT___GEN = 0ULL;
        __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN = 0ULL;
    } else {
        if (((IData)(vlSelfRef.address_command__DOT____Vcellinp__Counter__increment) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__Counter__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN));
        }
        if (((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__EmitEveryNImpl_6__DOT__ControlReg__DOT__state) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN));
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_0.address_command.pipelineStage_0: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_0.address_command.pipelineStage_2: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_0.address_command.pipelineStage_4: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__address_command__DOT__start_addr = 0ULL;
        vlSelfRef.__PVT__address_command__DOT__flits_total = 0ULL;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
             & (0x00000010U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_msg_reg[2U]))) {
            vlSelfRef.__PVT__address_command__DOT__start_addr 
                = (((QData)((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_msg_reg[1U])) 
                    << 0x00000020U) | (QData)((IData)(
                                                      vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_msg_reg[0U])));
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
             & (0x00000018U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_msg_reg[2U]))) {
            vlSelfRef.__PVT__address_command__DOT__flits_total 
                = (((QData)((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_msg_reg[1U])) 
                    << 0x00000020U) | (QData)((IData)(
                                                      vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_msg_reg[0U])));
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_0.address_command.pipelineStage_3: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_0.address_command.pipelineStage_1: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
            __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l;
            __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_0.address_command.pipelineStage: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
                __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
                __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1 = 0ULL;
    } else if (vlSelfRef.__PVT__address_command__DOT___GEN_1) {
        vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1 
            = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    }
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
        = __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
        = __Vdly__address_command__DOT__Counter__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg)));
    vlSelfRef.address_command__DOT____Vcellinp__Counter__increment 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__TaggedWriteGearboxImpl_6__DOT__ControlReg__DOT__state)) 
           & ((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
              & (vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
                 < vlSelfRef.__PVT__address_command__DOT__flits_total)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready) 
           & ((~ (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                  >> 3U)) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100000__DOT__inp_valid)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
           & ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100000__DOT__inp_valid) 
              & (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                 >> 3U)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready) 
           & ((~ (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                  >> 3U)) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100001__DOT__inp_valid)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
        = __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT___GEN_1 = 
        (vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
         == vlSelfRef.__PVT__address_command__DOT__flits_total);
}

void VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0__1(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__address_command__DOT___GEN = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_valid_reg__DOT__state) 
                                                   & ((IData)(vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready) 
                                                      & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_msg_reg[3U]));
    vlSelfRef.addrCmdCycles_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_61__DOT__output_channel_ready));
    vlSelfRef.addrCmdResponses_data_ready = ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg)) 
                                             & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_60__DOT__output_channel_ready));
    vlSelfRef.addrCmdIssued_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_60__DOT__output_channel_ready));
    vlSelfRef.__PVT__address_command__DOT__command_go 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
           & (0x00000020U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_msg_reg[2U]));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdCycles_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdResponses_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdIssued_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg));
}

void VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1__0(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    QData/*63:0*/ __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN = 0;
    QData/*63:0*/ __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN = 0;
    QData/*63:0*/ __Vdly__address_command__DOT__Counter__DOT___GEN;
    __Vdly__address_command__DOT__Counter__DOT___GEN = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 0;
    // Body
    __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    __Vdly__address_command__DOT__Counter__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid;
    __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i16__DOT__out1_valid_reg__DOT__state)) 
           & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_28_0__DOT__output_channel_ready));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__Counter__DOT___GEN = 0U;
        __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN = 0ULL;
    } else {
        vlSelfRef.__PVT__Counter__DOT___GEN = ((IData)(1U) 
                                               + vlSelfRef.__PVT__Counter__DOT___GEN);
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN));
        }
    }
    vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state 
        = ((~ (IData)(vlSymsp->TOP.rst)) & ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go) 
                                            | ((~ (IData)(vlSelfRef.__PVT__address_command__DOT___GEN_1)) 
                                               & (IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state))));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdCycles_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdIssued_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdResponses_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready));
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__Counter__DOT___GEN = 0ULL;
        __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN = 0ULL;
    } else {
        if (((IData)(vlSelfRef.address_command__DOT____Vcellinp__Counter__increment) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__Counter__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN));
        }
        if (((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__EmitEveryNImpl_7__DOT__ControlReg__DOT__state) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN));
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_1.address_command.pipelineStage_0: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_1.address_command.pipelineStage_2: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_1.address_command.pipelineStage_4: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__address_command__DOT__start_addr = 0ULL;
        vlSelfRef.__PVT__address_command__DOT__flits_total = 0ULL;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
             & (0x00000010U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_msg_reg[2U]))) {
            vlSelfRef.__PVT__address_command__DOT__start_addr 
                = (((QData)((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_msg_reg[1U])) 
                    << 0x00000020U) | (QData)((IData)(
                                                      vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_msg_reg[0U])));
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
             & (0x00000018U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_msg_reg[2U]))) {
            vlSelfRef.__PVT__address_command__DOT__flits_total 
                = (((QData)((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_msg_reg[1U])) 
                    << 0x00000020U) | (QData)((IData)(
                                                      vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_msg_reg[0U])));
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_1.address_command.pipelineStage_1: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_1.address_command.pipelineStage_3: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
            __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l;
            __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_1.address_command.pipelineStage: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
                __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
                __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1 = 0ULL;
    } else if (vlSelfRef.__PVT__address_command__DOT___GEN_1) {
        vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1 
            = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    }
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
        = __Vdly__address_command__DOT__Counter__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
        = __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg)));
    vlSelfRef.address_command__DOT____Vcellinp__Counter__increment 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__TaggedWriteGearboxImpl_7__DOT__ControlReg__DOT__state)) 
           & ((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
              & (vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
                 < vlSelfRef.__PVT__address_command__DOT__flits_total)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready) 
           & ((~ (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                  >> 3U)) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100101__DOT__inp_valid)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
           & ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100101__DOT__inp_valid) 
              & (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                 >> 3U)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready) 
           & ((~ (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                  >> 3U)) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100110__DOT__inp_valid)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
        = __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT___GEN_1 = 
        (vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
         == vlSelfRef.__PVT__address_command__DOT__flits_total);
}

void VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1__1(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__address_command__DOT___GEN = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_valid_reg__DOT__state) 
                                                   & ((IData)(vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready) 
                                                      & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_msg_reg[3U]));
    vlSelfRef.addrCmdIssued_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_72__DOT__output_channel_ready));
    vlSelfRef.addrCmdCycles_data_ready = ((~ (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1.__PVT__pipelineStage__DOT__x_valid_reg)) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_69__DOT__output_channel_ready));
    vlSelfRef.addrCmdResponses_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg) 
                                             & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_71__DOT__output_channel_ready));
    vlSelfRef.__PVT__address_command__DOT__command_go 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
           & (0x00000020U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_msg_reg[2U]));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdIssued_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdCycles_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdResponses_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg));
}

void VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2__0(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    QData/*63:0*/ __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN = 0;
    QData/*63:0*/ __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN = 0;
    QData/*63:0*/ __Vdly__address_command__DOT__Counter__DOT___GEN;
    __Vdly__address_command__DOT__Counter__DOT___GEN = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 0;
    CData/*0:0*/ __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 0;
    // Body
    __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    __Vdly__address_command__DOT__Counter__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid;
    __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
        = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage__DOT__l_valid 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid;
    __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg 
        = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__Counter__DOT___GEN = 0U;
        __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN = 0ULL;
    } else {
        vlSelfRef.__PVT__Counter__DOT___GEN = ((IData)(1U) 
                                               + vlSelfRef.__PVT__Counter__DOT___GEN);
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN));
        }
    }
    vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state 
        = ((~ (IData)(vlSymsp->TOP.rst)) & ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go) 
                                            | ((~ (IData)(vlSelfRef.__PVT__address_command__DOT___GEN_1)) 
                                               & (IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state))));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdCycles_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdIssued_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.addrCmdResponses_data_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg 
        = ((~ (IData)(vlSymsp->TOP.rst)) & (IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready));
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__Counter__DOT___GEN = 0ULL;
        __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN = 0ULL;
    } else {
        if (((IData)(vlSelfRef.address_command__DOT____Vcellinp__Counter__increment) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__Counter__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN));
        }
        if (((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__EmitEveryNImpl_8__DOT__ControlReg__DOT__state) 
             | (IData)(vlSelfRef.__PVT__address_command__DOT__command_go))) {
            __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
                = ((IData)(vlSelfRef.__PVT__address_command__DOT__command_go)
                    ? 0ULL : (1ULL + vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN));
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_2.address_command.pipelineStage_0: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_2.address_command.pipelineStage_2: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)))) {
            __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid))) {
            __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_2.address_command.pipelineStage_4: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg) {
                __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid = 1U;
            } else {
                __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__address_command__DOT__start_addr = 0ULL;
    } else if (((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
                & (0x00000010U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i19__DOT__out0_msg_reg[2U]))) {
        vlSelfRef.__PVT__address_command__DOT__start_addr 
            = (((QData)((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i19__DOT__out0_msg_reg[1U])) 
                << 0x00000020U) | (QData)((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i19__DOT__out0_msg_reg[0U])));
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_2.address_command.pipelineStage_3: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_2.address_command.pipelineStage_1: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN;
                __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__address_command__DOT__flits_total = 0ULL;
    } else if (((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
                & (0x00000018U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i19__DOT__out0_msg_reg[2U]))) {
        vlSelfRef.__PVT__address_command__DOT__flits_total 
            = (((QData)((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i19__DOT__out0_msg_reg[1U])) 
                << 0x00000020U) | (QData)((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i19__DOT__out0_msg_reg[0U])));
    }
    if (vlSymsp->TOP.rst) {
        __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 0U;
        __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 0U;
    } else {
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit) 
             & (~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
            __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit;
        }
        if (((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit) 
             & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid))) {
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l;
            __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 1U;
            __Vdly__address_command__DOT__pipelineStage__DOT__l_valid 
                = vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit;
            vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l 
                = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
        } else if (((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit)) 
                    & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit))) {
            if (vlSymsp->_vm_contextp__->assertOnGet(2, 1)) {
                if (VL_UNLIKELY((vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid))) {
                    VL_WRITEF_NX("[%0t] %%Error: ESIPrimitives.sv:107: Assertion failed in %NESI_Cosim_Top.ESI_Cosim_UserTopWrapper.EsiTester.writemem_2.address_command.pipelineStage: 'assert' failed.\n",0,
                                 64,VL_TIME_UNITED_Q(1),
                                 -12,vlSymsp->name());
                    VL_STOP_MT("/tmp/esi-pytest-class-compile-wf_dhfj4/hw/ESIPrimitives.sv", 107, "");
                }
            }
            if (vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg) {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
                __Vdly__address_command__DOT__pipelineStage__DOT__l_valid = 1U;
            } else {
                vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_reg 
                    = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
                __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg = 1U;
            }
        }
    }
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1 = 0ULL;
    } else if (vlSelfRef.__PVT__address_command__DOT___GEN_1) {
        vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycles___05Freg1 
            = vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    }
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_0__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_2__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_4__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_3__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
        = __Vdly__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage_1__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
        = __Vdly__address_command__DOT__Counter__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid 
        = __Vdly__address_command__DOT__pipelineStage__DOT__l_valid;
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg 
        = __Vdly__address_command__DOT__pipelineStage__DOT__x_valid_reg;
    vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg)));
    vlSelfRef.address_command__DOT____Vcellinp__Counter__increment 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__TaggedWriteGearboxImpl_8__DOT__ControlReg__DOT__state)) 
           & ((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
              & (vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
                 < vlSelfRef.__PVT__address_command__DOT__flits_total)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready) 
           & ((~ (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                  >> 3U)) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path101010__DOT__inp_valid)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
           & ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path101010__DOT__inp_valid) 
              & (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                 >> 3U)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready) 
           & ((~ (vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_msg_reg[2U] 
                  >> 3U)) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path101011__DOT__inp_valid)));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN 
        = __Vdly__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
    vlSelfRef.__PVT__address_command__DOT___GEN_1 = 
        (vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
         == vlSelfRef.__PVT__address_command__DOT__flits_total);
}

void VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2__1(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i18__DOT__out1_valid_reg__DOT__state)) 
           & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_35_0__DOT__output_channel_ready));
    vlSelfRef.addrCmdCycles_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_81__DOT__output_channel_ready));
    vlSelfRef.addrCmdResponses_data_ready = ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg)) 
                                             & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_80__DOT__output_channel_ready));
    vlSelfRef.addrCmdIssued_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_80__DOT__output_channel_ready));
    vlSelfRef.__PVT__address_command__DOT___GEN = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i19__DOT__out0_valid_reg__DOT__state) 
                                                   & ((IData)(vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready) 
                                                      & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i19__DOT__out0_msg_reg[3U]));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdCycles_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdResponses_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdIssued_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__command_go 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
           & (0x00000020U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i19__DOT__out0_msg_reg[2U]));
}
