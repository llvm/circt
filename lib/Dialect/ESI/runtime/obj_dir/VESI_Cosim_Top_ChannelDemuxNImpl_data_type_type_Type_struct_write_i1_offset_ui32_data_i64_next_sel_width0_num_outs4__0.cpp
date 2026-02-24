// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VESI_Cosim_Top.h for the primary calling header

#include "VESI_Cosim_Top__pch.h"

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i0__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i0__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_1) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_1)) 
                                                         & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_2) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_2)) 
                                                         & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_3) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_3)) 
                                                         & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_0) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_0)) 
                                                         & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state))));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__out0_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[3U] = 0U;
    } else {
        if (vlSelfRef.__PVT__will_write_0) {
            vlSelfRef.__PVT__out0_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out0_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out0_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out0_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_2) {
            vlSelfRef.__PVT__out2_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out2_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out2_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out2_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_3) {
            vlSelfRef.__PVT__out3_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out3_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out3_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out3_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_1) {
            vlSelfRef.__PVT__out1_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out1_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out1_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out1_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U]);
        }
    }
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__cb_test__DOT__cmd_cmd_ready));
    vlSelfRef.__PVT__consume_2 = vlSelfRef.__PVT__out2_valid_reg__DOT__state;
    vlSelfRef.__PVT__consume_3 = vlSelfRef.__PVT__out3_valid_reg__DOT__state;
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i0__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i0__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_0__DOT__output_channel_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i1__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i1__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_3) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_3)) 
                                                         & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_2) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_2)) 
                                                         & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_0) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_0)) 
                                                         & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_1) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_1)) 
                                                         & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state))));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__out0_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[3U] = 0U;
    } else {
        if (vlSelfRef.__PVT__will_write_0) {
            vlSelfRef.__PVT__out0_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out0_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out0_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out0_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_3) {
            vlSelfRef.__PVT__out3_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out3_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out3_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out3_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_1) {
            vlSelfRef.__PVT__out1_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out1_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out1_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out1_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_2) {
            vlSelfRef.__PVT__out2_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out2_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out2_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out2_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U]);
        }
    }
    vlSelfRef.__PVT__consume_0 = vlSelfRef.__PVT__out0_valid_reg__DOT__state;
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i1__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i1__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
    vlSelfRef.output_3_ready = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_6_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_32__DOT__address_command__DOT__cmd_32_data_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_32__DOT__address_command__DOT__cmd_32_data_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i2__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i2__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_1) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_1)) 
                                                         & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_3) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_3)) 
                                                         & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_0) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_0)) 
                                                         & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_2) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_2)) 
                                                         & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state))));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__out3_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[3U] = 0U;
    } else {
        if (vlSelfRef.__PVT__will_write_3) {
            vlSelfRef.__PVT__out3_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[0U];
            vlSelfRef.__PVT__out3_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[1U];
            vlSelfRef.__PVT__out3_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[2U];
            vlSelfRef.__PVT__out3_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_0) {
            vlSelfRef.__PVT__out0_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[0U];
            vlSelfRef.__PVT__out0_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[1U];
            vlSelfRef.__PVT__out0_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[2U];
            vlSelfRef.__PVT__out0_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_1) {
            vlSelfRef.__PVT__out1_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[0U];
            vlSelfRef.__PVT__out1_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[1U];
            vlSelfRef.__PVT__out1_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[2U];
            vlSelfRef.__PVT__out1_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_2) {
            vlSelfRef.__PVT__out2_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[0U];
            vlSelfRef.__PVT__out2_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[1U];
            vlSelfRef.__PVT__out2_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[2U];
            vlSelfRef.__PVT__out2_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U]);
        }
    }
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i2__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i2__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_2) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_2) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_2) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_2) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U])));
    vlSelfRef.output_3_ready = ((~ (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state)) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_9_0__DOT__output_channel_ready));
    vlSelfRef.output_0_ready = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_5_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_64__DOT__address_command__DOT__cmd_64_data_ready));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_64__DOT__address_command__DOT__cmd_64_data_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i3__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i3__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_2) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_2)) 
                                                         & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_3) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_3)) 
                                                         & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_1) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_1)) 
                                                         & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_0) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_0)) 
                                                         & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state))));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__out3_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[3U] = 0U;
    } else {
        if (vlSelfRef.__PVT__will_write_3) {
            vlSelfRef.__PVT__out3_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[0U];
            vlSelfRef.__PVT__out3_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[1U];
            vlSelfRef.__PVT__out3_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[2U];
            vlSelfRef.__PVT__out3_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_0) {
            vlSelfRef.__PVT__out0_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[0U];
            vlSelfRef.__PVT__out0_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[1U];
            vlSelfRef.__PVT__out0_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[2U];
            vlSelfRef.__PVT__out0_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_2) {
            vlSelfRef.__PVT__out2_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[0U];
            vlSelfRef.__PVT__out2_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[1U];
            vlSelfRef.__PVT__out2_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[2U];
            vlSelfRef.__PVT__out2_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_1) {
            vlSelfRef.__PVT__out1_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[0U];
            vlSelfRef.__PVT__out1_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[1U];
            vlSelfRef.__PVT__out1_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[2U];
            vlSelfRef.__PVT__out1_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U]);
        }
    }
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i3__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i3__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_3) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_3) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_3) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_3) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U])));
    vlSelfRef.output_0_ready = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_11_0__DOT__output_channel_ready));
    vlSelfRef.output_3_ready = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_14_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_128__DOT__address_command__DOT__cmd_128_data_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_128__DOT__address_command__DOT__cmd_128_data_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i4__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i4__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_3) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_3)) 
                                                         & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_2) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_2)) 
                                                         & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_1) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_1)) 
                                                         & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_0) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_0)) 
                                                         & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state))));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__out3_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[3U] = 0U;
    } else {
        if (vlSelfRef.__PVT__will_write_3) {
            vlSelfRef.__PVT__out3_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out3_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out3_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out3_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_0) {
            vlSelfRef.__PVT__out0_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out0_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out0_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out0_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_2) {
            vlSelfRef.__PVT__out2_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out2_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out2_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out2_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_1) {
            vlSelfRef.__PVT__out1_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out1_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out1_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out1_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U]);
        }
    }
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i4__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i4__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_0) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_0) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_0) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_0) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U])));
    vlSelfRef.output_0_ready = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_13_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_256__DOT__address_command__DOT__cmd_256_data_ready));
    vlSelfRef.output_3_ready = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_15_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_256__DOT__address_command__DOT__cmd_256_data_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_0) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_0)) 
                                                         & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_3) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_3)) 
                                                         & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_2) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_2)) 
                                                         & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_1) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_1)) 
                                                         & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state))));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__out3_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[3U] = 0U;
    } else {
        if (vlSelfRef.__PVT__will_write_3) {
            vlSelfRef.__PVT__out3_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out3_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out3_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out3_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_0) {
            vlSelfRef.__PVT__out0_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out0_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out0_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out0_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_2) {
            vlSelfRef.__PVT__out2_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out2_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out2_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out2_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_1) {
            vlSelfRef.__PVT__out1_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out1_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out1_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out1_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U]);
        }
    }
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
    vlSelfRef.output_3_ready = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_22_0__DOT__output_channel_ready));
    vlSelfRef.output_0_ready = ((~ (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i4.__PVT__out3_valid_reg__DOT__state)) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_15_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5__2(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5__2\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512.__PVT__address_command__DOT__cmd_512_data_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i6__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i6__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_1) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_1)) 
                                                         & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_3) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_3)) 
                                                         & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_2) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_2)) 
                                                         & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_0) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_0)) 
                                                         & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state))));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__out3_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[3U] = 0U;
    } else {
        if (vlSelfRef.__PVT__will_write_3) {
            vlSelfRef.__PVT__out3_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[0U];
            vlSelfRef.__PVT__out3_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[1U];
            vlSelfRef.__PVT__out3_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[2U];
            vlSelfRef.__PVT__out3_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_0) {
            vlSelfRef.__PVT__out0_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[0U];
            vlSelfRef.__PVT__out0_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[1U];
            vlSelfRef.__PVT__out0_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[2U];
            vlSelfRef.__PVT__out0_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_2) {
            vlSelfRef.__PVT__out2_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[0U];
            vlSelfRef.__PVT__out2_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[1U];
            vlSelfRef.__PVT__out2_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[2U];
            vlSelfRef.__PVT__out2_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_1) {
            vlSelfRef.__PVT__out1_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[0U];
            vlSelfRef.__PVT__out1_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[1U];
            vlSelfRef.__PVT__out1_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[2U];
            vlSelfRef.__PVT__out1_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U]);
        }
    }
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i6__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i6__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_2) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_2) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_2) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_2) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U])));
    vlSelfRef.output_0_ready = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_21_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_534__DOT__address_command__DOT__cmd_534_data_ready));
    vlSelfRef.output_3_ready = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_24_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_534__DOT__address_command__DOT__cmd_534_data_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_3) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_3)) 
                                                         & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_2) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_2)) 
                                                         & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_1) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_1)) 
                                                         & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_0) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_0)) 
                                                         & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state))));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__out3_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[3U] = 0U;
    } else {
        if (vlSelfRef.__PVT__will_write_3) {
            vlSelfRef.__PVT__out3_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[0U];
            vlSelfRef.__PVT__out3_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[1U];
            vlSelfRef.__PVT__out3_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[2U];
            vlSelfRef.__PVT__out3_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_0) {
            vlSelfRef.__PVT__out0_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[0U];
            vlSelfRef.__PVT__out0_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[1U];
            vlSelfRef.__PVT__out0_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[2U];
            vlSelfRef.__PVT__out0_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_2) {
            vlSelfRef.__PVT__out2_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[0U];
            vlSelfRef.__PVT__out2_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[1U];
            vlSelfRef.__PVT__out2_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[2U];
            vlSelfRef.__PVT__out2_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_1) {
            vlSelfRef.__PVT__out1_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[0U];
            vlSelfRef.__PVT__out1_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[1U];
            vlSelfRef.__PVT__out1_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[2U];
            vlSelfRef.__PVT__out1_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U]);
        }
    }
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
    vlSelfRef.output_0_ready = ((~ (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i6.__PVT__out3_valid_reg__DOT__state)) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_24_0__DOT__output_channel_ready));
    vlSelfRef.output_3_ready = ((~ (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state)) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_26_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7__2(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7__2\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_0.__PVT__address_command__DOT__cmd_512_data_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_3) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_3)) 
                                                         & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_2) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_2)) 
                                                         & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_1) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_1)) 
                                                         & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_0) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_0)) 
                                                         & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state))));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__out3_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[3U] = 0U;
    } else {
        if (vlSelfRef.__PVT__will_write_3) {
            vlSelfRef.__PVT__out3_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out3_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out3_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out3_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_0) {
            vlSelfRef.__PVT__out0_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out0_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out0_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out0_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_2) {
            vlSelfRef.__PVT__out2_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out2_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out2_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out2_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_1) {
            vlSelfRef.__PVT__out1_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[0U];
            vlSelfRef.__PVT__out1_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[1U];
            vlSelfRef.__PVT__out1_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[2U];
            vlSelfRef.__PVT__out1_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U]);
        }
    }
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
    vlSelfRef.output_0_ready = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_30_0__DOT__output_channel_ready));
    vlSelfRef.output_3_ready = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_31_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8__2(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8__2\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1.__PVT__address_command__DOT__cmd_512_data_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_3) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_3)) 
                                                         & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_0) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_0)) 
                                                         & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_2) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_2)) 
                                                         & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_1) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_1)) 
                                                         & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state))));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__out3_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out3_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[3U] = 0U;
    } else {
        if (vlSelfRef.__PVT__will_write_3) {
            vlSelfRef.__PVT__out3_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out3_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out3_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out3_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_0) {
            vlSelfRef.__PVT__out0_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out0_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out0_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out0_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_2) {
            vlSelfRef.__PVT__out2_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out2_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out2_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out2_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_1) {
            vlSelfRef.__PVT__out1_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[0U];
            vlSelfRef.__PVT__out1_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[1U];
            vlSelfRef.__PVT__out1_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[2U];
            vlSelfRef.__PVT__out1_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U]);
        }
    }
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
    vlSelfRef.output_0_ready = ((~ (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8.__PVT__out3_valid_reg__DOT__state)) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_31_0__DOT__output_channel_ready));
    vlSelfRef.output_3_ready = ((~ (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state)) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_34_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
    vlSelfRef.__PVT__consume_3 = ((IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_3_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9__2(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9__2\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_2.__PVT__address_command__DOT__cmd_512_data_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2.__PVT__address_command__DOT__cmd_512_data_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i10__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i10__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & (IData)(vlSelfRef.__PVT__will_write_3));
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_2) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_2)) 
                                                         & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_1) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_1)) 
                                                         & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state))));
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & ((IData)(vlSelfRef.__PVT__will_write_0) 
                                                      | ((~ (IData)(vlSelfRef.__PVT__consume_0)) 
                                                         & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state))));
    if (vlSymsp->TOP.rst) {
        vlSelfRef.__PVT__out1_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out1_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out0_msg_reg[3U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[0U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[1U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[2U] = 0U;
        vlSelfRef.__PVT__out2_msg_reg[3U] = 0U;
    } else {
        if (vlSelfRef.__PVT__will_write_1) {
            vlSelfRef.__PVT__out1_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[0U];
            vlSelfRef.__PVT__out1_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[1U];
            vlSelfRef.__PVT__out1_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[2U];
            vlSelfRef.__PVT__out1_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_0) {
            vlSelfRef.__PVT__out0_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[0U];
            vlSelfRef.__PVT__out0_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[1U];
            vlSelfRef.__PVT__out0_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[2U];
            vlSelfRef.__PVT__out0_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U]);
        }
        if (vlSelfRef.__PVT__will_write_2) {
            vlSelfRef.__PVT__out2_msg_reg[0U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[0U];
            vlSelfRef.__PVT__out2_msg_reg[1U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[1U];
            vlSelfRef.__PVT__out2_msg_reg[2U] = vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[2U];
            vlSelfRef.__PVT__out2_msg_reg[3U] = (1U 
                                                 & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U]);
        }
    }
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i10__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i10__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_1 = ((IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state) 
                                  & (((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                          >> 9U)) & 
                                      (((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                            >> 8U)) 
                                        & (((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                >> 7U)) 
                                            & (((~ 
                                                 (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                  >> 6U)) 
                                                & (((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_32__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_32__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_32__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path000001__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_32__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_32__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_32__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path000011__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U)))) 
                                               | ((((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path000100__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path000100__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_64__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_64__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_64__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path000110__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_64__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_64__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U))) 
                                                  & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                     >> 6U)))) 
                                           | ((((~ 
                                                 (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                  >> 6U)) 
                                                & (((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_64__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001000__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001001__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001001__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_128__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_128__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_128__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001011__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U)))) 
                                               | ((((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_128__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_128__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_128__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001101__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001110__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path001110__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_256__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_256__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U))) 
                                                  & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                     >> 6U))) 
                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                 >> 7U)))) 
                                       | ((((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                >> 7U)) 
                                            & (((~ 
                                                 (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                  >> 6U)) 
                                                & (((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_256__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010000__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_256__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_256__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_256__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010010__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010011__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010011__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U)))) 
                                               | ((((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_512.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010101__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_512.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path010111__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U))) 
                                                  & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                     >> 6U)))) 
                                           | ((((~ 
                                                 (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                  >> 6U)) 
                                                & (((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011000__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011000__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_534__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_534__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_534__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011010__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_534__DOT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_534__DOT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U)))) 
                                               | ((((~ 
                                                     (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                      >> 5U)) 
                                                    & (((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_534__DOT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011100__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011101__DOT__output0_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011101__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U)))) 
                                                   | ((((~ 
                                                         (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                          >> 4U)) 
                                                        & (((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_0.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_0.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U)))) 
                                                       | ((((~ 
                                                             (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                              >> 3U)) 
                                                            & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_0.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                           | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path011111__DOT__output1_ready) 
                                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                 >> 3U))) 
                                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                             >> 4U))) 
                                                      & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                         >> 5U))) 
                                                  & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                     >> 6U))) 
                                              & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                 >> 7U))) 
                                          & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                             >> 8U)))) 
                                     | ((((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                              >> 8U)) 
                                          & (((~ (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                  >> 7U)) 
                                              & (((~ 
                                                   (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                    >> 6U)) 
                                                  & (((~ 
                                                       (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                        >> 5U)) 
                                                      & (((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_0.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100001__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U)))) 
                                                     | ((((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100010__DOT__output0_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100010__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U))) 
                                                        & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                           >> 5U)))) 
                                                 | ((((~ 
                                                       (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                        >> 5U)) 
                                                      & (((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_1.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100100__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U)))) 
                                                     | ((((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_1.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100110__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100111__DOT__output0_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path100111__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U))) 
                                                        & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                           >> 5U))) 
                                                    & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                       >> 6U)))) 
                                             | ((((~ 
                                                   (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                    >> 6U)) 
                                                  & (((~ 
                                                       (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                        >> 5U)) 
                                                      & (((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_2.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_2.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__readmem_2.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path101001__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U)))) 
                                                     | ((((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2.__PVT__address_command__DOT__addrCmdCycles_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2.__PVT__address_command__DOT__addrCmdIssued_get_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | ((((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__EsiTester__DOT__writemem_2.__PVT__address_command__DOT__addrCmdResponses_get_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path101011__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U))) 
                                                            & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                               >> 4U))) 
                                                        & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                           >> 5U)))) 
                                                 | ((((~ 
                                                       (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                        >> 5U)) 
                                                      & (((~ 
                                                           (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U)) 
                                                          & (((~ 
                                                               (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                >> 3U)) 
                                                              & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path101100__DOT__output0_ready)) 
                                                             | ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__telemetry_client_cmd_demux__DOT__demux2_path101100__DOT__output1_ready) 
                                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                                   >> 3U)))) 
                                                         | (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                            >> 4U))) 
                                                     | (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                        >> 5U)) 
                                                    & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                       >> 6U))) 
                                                & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                                   >> 7U)))) 
                                         | (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                            >> 8U)) 
                                        & (vlSelfRef.__PVT__out1_msg_reg[2U] 
                                           >> 9U))));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
    vlSelfRef.output_0_ready = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_37_0__DOT__output_channel_ready));
    vlSelfRef.__PVT__consume_2 = ((IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state) 
                                  & ((~ (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_88__DOT__output_channel_valid)) 
                                     & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__ChannelMux2_36_0__DOT__output_channel_ready)));
    vlSelfRef.__PVT__consume_0 = ((IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state) 
                                  & (IData)(vlSelfRef.output_0_ready));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i11__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i11__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__out0_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & (IData)(vlSelfRef.__PVT__will_write_0));
    vlSelfRef.__PVT__out1_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & (IData)(vlSelfRef.__PVT__will_write_1));
    vlSelfRef.__PVT__out2_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & (IData)(vlSelfRef.__PVT__will_write_2));
    vlSelfRef.__PVT__out3_valid_reg__DOT__state = (
                                                   (~ (IData)(vlSymsp->TOP.rst)) 
                                                   & (IData)(vlSelfRef.__PVT__will_write_3));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i11__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i11__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_3) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_3) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_3) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_3) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out3_msg_reg[3U])));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i12__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i12__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_0) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_0) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_0) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_0) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out0_msg_reg[3U])));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i13__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i13__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_1) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_1) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_1) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_1) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out1_msg_reg[3U])));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i14__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i14__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_2) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_2) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_2) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_2) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out2_msg_reg[3U])));
}

void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i15__1(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width0_num_outs4___nba_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i15__1\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_3) 
                                     & (0U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_3) 
                                     & (2U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_3) 
                                     & (4U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_3) 
                                     & (6U == (6U & 
                                               vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out3_msg_reg[3U])));
}
