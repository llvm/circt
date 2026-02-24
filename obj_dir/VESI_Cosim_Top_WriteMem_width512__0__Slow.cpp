// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VESI_Cosim_Top.h for the primary calling header

#include "VESI_Cosim_Top__pch.h"

VL_ATTR_COLD void VESI_Cosim_Top_WriteMem_width512___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512__0(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__address_command__DOT___GEN_1 = 
        (vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
         == vlSelfRef.__PVT__address_command__DOT__flits_total);
    vlSelfRef.address_command__DOT____Vcellinp__Counter__increment 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__TaggedWriteGearboxImpl_4__DOT__ControlReg__DOT__state)) 
           & ((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
              & (vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
                 < vlSelfRef.__PVT__address_command__DOT__flits_total)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i10__DOT__out1_valid_reg__DOT__state)) 
           & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_19_0__DOT__output_channel_ready));
    vlSelfRef.addrCmdResponses_data_ready = ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg)) 
                                             & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_44__DOT__output_channel_ready));
    vlSelfRef.addrCmdIssued_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_44__DOT__output_channel_ready));
    vlSelfRef.addrCmdCycles_data_ready = ((~ (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512.__PVT__pipelineStage__DOT__x_valid_reg)) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_38__DOT__output_channel_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg));
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
    vlSelfRef.__PVT__address_command__DOT___GEN = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_valid_reg__DOT__state) 
                                                   & ((IData)(vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready) 
                                                      & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_msg_reg[3U]));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdResponses_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdIssued_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdCycles_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__command_go 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
           & (0x00000020U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_msg_reg[2U]));
}

VL_ATTR_COLD void VESI_Cosim_Top_WriteMem_width512___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0__0(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__address_command__DOT___GEN_1 = 
        (vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
         == vlSelfRef.__PVT__address_command__DOT__flits_total);
    vlSelfRef.address_command__DOT____Vcellinp__Counter__increment 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__TaggedWriteGearboxImpl_6__DOT__ControlReg__DOT__state)) 
           & ((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
              & (vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
                 < vlSelfRef.__PVT__address_command__DOT__flits_total)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready 
        = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_valid_reg__DOT__state) 
           & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_27_0__DOT__output_channel_ready));
    vlSelfRef.addrCmdCycles_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_61__DOT__output_channel_ready));
    vlSelfRef.addrCmdResponses_data_ready = ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg)) 
                                             & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_60__DOT__output_channel_ready));
    vlSelfRef.addrCmdIssued_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_60__DOT__output_channel_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg));
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
    vlSelfRef.__PVT__address_command__DOT___GEN = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_valid_reg__DOT__state) 
                                                   & ((IData)(vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready) 
                                                      & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_msg_reg[3U]));
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
           & (0x00000020U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_msg_reg[2U]));
}

VL_ATTR_COLD void VESI_Cosim_Top_WriteMem_width512___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1__0(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__address_command__DOT___GEN_1 = 
        (vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
         == vlSelfRef.__PVT__address_command__DOT__flits_total);
    vlSelfRef.address_command__DOT____Vcellinp__Counter__increment 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__TaggedWriteGearboxImpl_7__DOT__ControlReg__DOT__state)) 
           & ((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
              & (vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
                 < vlSelfRef.__PVT__address_command__DOT__flits_total)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i16__DOT__out1_valid_reg__DOT__state)) 
           & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_28_0__DOT__output_channel_ready));
    vlSelfRef.addrCmdIssued_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_72__DOT__output_channel_ready));
    vlSelfRef.addrCmdCycles_data_ready = ((~ (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1.__PVT__pipelineStage__DOT__x_valid_reg)) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_69__DOT__output_channel_ready));
    vlSelfRef.addrCmdResponses_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg) 
                                             & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_71__DOT__output_channel_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg));
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
    vlSelfRef.__PVT__address_command__DOT___GEN = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_valid_reg__DOT__state) 
                                                   & ((IData)(vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready) 
                                                      & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_msg_reg[3U]));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdIssued_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdCycles_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__xmit 
        = ((IData)(vlSelfRef.addrCmdResponses_data_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__command_go 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___GEN) 
           & (0x00000020U == vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_msg_reg[2U]));
}

VL_ATTR_COLD void VESI_Cosim_Top_WriteMem_width512___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2__0(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__address_command__DOT___GEN_1 = 
        (vlSelfRef.__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN 
         == vlSelfRef.__PVT__address_command__DOT__flits_total);
    vlSelfRef.address_command__DOT____Vcellinp__Counter__increment 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__HostMemWriteProcessorImpl__DOT__TaggedWriteGearboxImpl_8__DOT__ControlReg__DOT__state)) 
           & ((IData)(vlSelfRef.__PVT__address_command__DOT__operation_active__DOT__state) 
              & (vlSelfRef.__PVT__address_command__DOT__Counter__DOT___GEN 
                 < vlSelfRef.__PVT__address_command__DOT__flits_total)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdCycles_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdIssued_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__addrCmdResponses_get_ready 
        = (1U & ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid)) 
                 | (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg)));
    vlSelfRef.__PVT__address_command__DOT__cmd_512_data_ready 
        = ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i18__DOT__out1_valid_reg__DOT__state)) 
           & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_35_0__DOT__output_channel_ready));
    vlSelfRef.addrCmdCycles_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_81__DOT__output_channel_ready));
    vlSelfRef.addrCmdResponses_data_ready = ((~ (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg)) 
                                             & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_80__DOT__output_channel_ready));
    vlSelfRef.addrCmdIssued_data_ready = ((IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg) 
                                          & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_80__DOT__output_channel_ready));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_1) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready_0) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg));
    vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__xmit 
        = ((IData)(vlSelfRef.__PVT__address_command__DOT___pipelineStage_a_ready) 
           & (IData)(vlSelfRef.__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg));
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

VL_ATTR_COLD void VESI_Cosim_Top_WriteMem_width512___ctor_var_reset(VESI_Cosim_Top_WriteMem_width512* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_WriteMem_width512___ctor_var_reset\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18209466448985614591ull);
    VL_SCOPED_RAND_RESET_W(97, vlSelf->cmd_512_cmd, __VscopeHash, 1869530427284923211ull);
    vlSelf->cmd_512_cmd_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16400032443104586953ull);
    vlSelf->addrCmdCycles_get_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2143341930126336946ull);
    vlSelf->addrCmdIssued_get_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4688673669784960362ull);
    vlSelf->addrCmdResponses_get_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 50948957598345308ull);
    vlSelf->host_ackTag = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 11723120092905643622ull);
    vlSelf->host_ackTag_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10112452129548494449ull);
    vlSelf->cmd_512_data_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 101163788524178052ull);
    vlSelf->addrCmdCycles_data_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9591382067523700782ull);
    vlSelf->addrCmdIssued_data_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18100921022683774558ull);
    vlSelf->addrCmdResponses_data_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14023819122466716088ull);
    vlSelf->host_req_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13339781475700046847ull);
    vlSelf->cmd_512_cmd_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10171841746444942880ull);
    vlSelf->addrCmdCycles_get_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1064860008935333416ull);
    vlSelf->addrCmdIssued_get_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1215012185373699824ull);
    vlSelf->addrCmdResponses_get_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15995680916885582060ull);
    vlSelf->host_ackTag_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10982167588913512458ull);
    vlSelf->cmd_512_data = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 12381199249687731242ull);
    vlSelf->cmd_512_data_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8349202351933488951ull);
    vlSelf->addrCmdCycles_data = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 11776758252848990958ull);
    vlSelf->addrCmdCycles_data_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14146692607762802000ull);
    vlSelf->addrCmdIssued_data = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 4300421506010391125ull);
    vlSelf->addrCmdIssued_data_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3858870809607468560ull);
    vlSelf->addrCmdResponses_data = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 9830272939956502044ull);
    vlSelf->addrCmdResponses_data_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13163076085514037736ull);
    VL_SCOPED_RAND_RESET_W(584, vlSelf->host_req, __VscopeHash, 2491196844537464692ull);
    vlSelf->host_req_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3431441902742231686ull);
    vlSelf->__PVT__Counter__DOT___GEN = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10544909457173139354ull);
    vlSelf->__PVT__address_command__DOT__cmd_512_data_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9079615466477236939ull);
    vlSelf->__PVT__address_command__DOT__addrCmdCycles_get_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4143645362231094488ull);
    vlSelf->__PVT__address_command__DOT__addrCmdIssued_get_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15582242218300784874ull);
    vlSelf->__PVT__address_command__DOT__addrCmdResponses_get_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11157817125289493601ull);
    vlSelf->__PVT__address_command__DOT__command_go = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6836492105907430906ull);
    vlSelf->__PVT__address_command__DOT___pipelineStage_a_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9972311127365593017ull);
    vlSelf->__PVT__address_command__DOT___pipelineStage_a_ready_0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16661660539098887483ull);
    vlSelf->__PVT__address_command__DOT___pipelineStage_a_ready_1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13610529995564224496ull);
    vlSelf->__PVT__address_command__DOT___GEN = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12619148439550438368ull);
    vlSelf->__PVT__address_command__DOT__start_addr = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 15066702375601128361ull);
    vlSelf->__PVT__address_command__DOT__flits_total = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 14305427765001343014ull);
    vlSelf->__PVT__address_command__DOT___GEN_1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12765135628126740819ull);
    vlSelf->__PVT__address_command__DOT__addr_cmd_cycles___05Freg1 = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 18197869137460487335ull);
    vlSelf->address_command__DOT____Vcellinp__Counter__increment = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16589053822940958527ull);
    vlSelf->__PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 13844233301026502046ull);
    vlSelf->__PVT__address_command__DOT__operation_active__DOT__state = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16559293975155672119ull);
    vlSelf->__PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 12576395147444790141ull);
    vlSelf->__PVT__address_command__DOT__Counter__DOT___GEN = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 1267997544680211527ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage__DOT__x_reg = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 14935732531040270186ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17739302595766900802ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 870995421634838306ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12571318214310723747ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage__DOT__l = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 14103722350402438276ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2064032619110434354ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12789062833509237031ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1647619452011438431ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_0__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1973426966323029831ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_0__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1129071341396329386ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9840591302034866316ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_1__DOT__x_reg = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 17853684568079665266ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5427580479766969110ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5452754484560851540ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_1__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2731506567251954667ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_1__DOT__l = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 6667962035281751167ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_1__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13753915588815980648ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15677537702085600087ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8730763206465220207ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_2__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10649648277241601469ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_2__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11229580623820756232ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15511085118650381435ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_3__DOT__x_reg = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 10790897181376919510ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16373324186395055420ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7984063750512138574ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_3__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9783396229293098610ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_3__DOT__l = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 1385020642454928244ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_3__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16136560450906726311ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16862336211324961425ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11802903949566072406ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_4__DOT__xmit = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11077293110030577478ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_4__DOT__l_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14475495134383060027ull);
    vlSelf->__PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14050754578767982339ull);
}
