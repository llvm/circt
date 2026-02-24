// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VESI_Cosim_Top.h for the primary calling header

#include "VESI_Cosim_Top__pch.h"

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i0__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i0__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i0__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i0__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i1__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i1__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i2__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i2__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i3__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i3__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_0) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out0_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i1__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i1__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i4__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i4__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i5__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i5__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i6__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i6__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i7__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i7__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_1) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out1_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i2__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i2__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i8__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i8__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i9__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i9__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i10__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i10__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i11__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_2) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_2) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_2) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_2) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out2_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i3__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i3__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i12__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i12__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i13__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i13__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i14__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i14__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i15__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_3) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_3) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_3) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__consume_3) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i0__DOT__out3_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i4__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i4__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i16__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i16__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i17__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i18__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i18__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i19__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i19__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_0) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_0) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_0) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_0) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out0_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i5__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i20__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i21__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i21__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i22__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i22__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i23__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i23__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_1) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out1_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i6__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i6__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i24__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i24__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i25__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i25__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i26__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i26__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i27__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i27__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_2) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_2) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_2) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_2) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out2_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i7__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i28__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i28__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i29__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i29__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i30__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i30__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i31__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i31__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__consume_3) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i1__DOT__out3_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i8__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i32__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i32__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i33__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i33__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i34__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i34__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i35__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i35__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_0) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out0_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i9__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i36__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i36__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i37__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i37__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i38__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i38__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i39__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i39__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_1) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out1_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i10__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i10__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i40__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i40__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i41__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i41__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i42__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i42__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i43__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i43__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_2) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out2_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i11__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i11__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i44__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i44__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i45__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i45__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i46__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i46__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i47__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i47__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_3) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_3) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_3) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__consume_3) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i2__DOT__out3_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i12__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i12__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i48__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i48__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i49__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i49__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i50__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i50__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i51__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i51__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_0) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_0) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_0) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out0_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_0) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out0_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i13__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i13__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i52__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i52__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i53__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i53__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i54__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i54__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i55__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i55__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_1) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_1) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_1) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out1_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_1) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out1_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i14__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i14__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i56__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i56__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i57__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i57__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i58__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i58__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i59__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i59__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_2) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_2) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_2) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out2_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_2) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out2_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i15__0(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___stl_sequent__TOP__ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l2_i15__0\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__PVT__consume_0 = ((~ (((~ (vlSelfRef.__PVT__out0_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i60__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out0_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i60__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out0_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_1 = ((~ (((~ (vlSelfRef.__PVT__out1_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i61__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out1_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i61__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out1_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_2 = ((~ (((~ (vlSelfRef.__PVT__out2_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i62__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out2_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i62__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out2_valid_reg__DOT__state));
    vlSelfRef.__PVT__consume_3 = ((~ (((~ (vlSelfRef.__PVT__out3_msg_reg[3U] 
                                           >> 1U)) 
                                       & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i63__DOT__out0_valid_reg__DOT__state)) 
                                      | ((vlSelfRef.__PVT__out3_msg_reg[3U] 
                                          >> 1U) & (IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l3_i63__DOT__out1_valid_reg__DOT__state)))) 
                                  & (IData)(vlSelfRef.__PVT__out3_valid_reg__DOT__state));
    vlSelfRef.__PVT__will_write_0 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_3) 
                                     & (0U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_1 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_3) 
                                     & (4U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_2 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_3) 
                                     & (8U == (0x0cU 
                                               & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out3_msg_reg[3U])));
    vlSelfRef.__PVT__will_write_3 = ((IData)(vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__consume_3) 
                                     & (0x0cU == (0x0cU 
                                                  & vlSymsp->TOP.ESI_Cosim_Top__DOT__ESI_Cosim_UserTopWrapper__DOT__client_cmd_demux__DOT__demux_l1_i3__DOT__out3_msg_reg[3U])));
}

VL_ATTR_COLD void VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___ctor_var_reset(VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VESI_Cosim_Top_ChannelDemuxNImpl_data_type_type_Type_struct_write_i1_offset_ui32_data_i64_next_sel_width1_num_outs4___ctor_var_reset\n"); );
    VESI_Cosim_Top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18209466448985614591ull);
    VL_SCOPED_RAND_RESET_W(100, vlSelf->inp, __VscopeHash, 11674450839837347981ull);
    vlSelf->inp_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9997336024544461226ull);
    vlSelf->output_0_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13248880085342039060ull);
    vlSelf->output_1_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7902298621141030093ull);
    vlSelf->output_2_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5285606803710789641ull);
    vlSelf->output_3_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4465621261820349703ull);
    vlSelf->inp_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17541595593609939190ull);
    VL_SCOPED_RAND_RESET_W(98, vlSelf->output_0, __VscopeHash, 3565184587078972990ull);
    vlSelf->output_0_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7386072530874344436ull);
    VL_SCOPED_RAND_RESET_W(98, vlSelf->output_1, __VscopeHash, 5600550167531168213ull);
    vlSelf->output_1_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11328874110889888298ull);
    VL_SCOPED_RAND_RESET_W(98, vlSelf->output_2, __VscopeHash, 13958066754333152748ull);
    vlSelf->output_2_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8143702230589480795ull);
    VL_SCOPED_RAND_RESET_W(98, vlSelf->output_3, __VscopeHash, 1839122939389980251ull);
    vlSelf->output_3_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18159941315985790452ull);
    vlSelf->__PVT__consume_3 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3851712065953137537ull);
    vlSelf->__PVT__consume_2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16535812106909167653ull);
    vlSelf->__PVT__consume_1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9423595460782033938ull);
    vlSelf->__PVT__consume_0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1656316240931420215ull);
    vlSelf->__PVT__will_write_0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16671588214008449789ull);
    VL_SCOPED_RAND_RESET_W(98, vlSelf->__PVT__out0_msg_reg, __VscopeHash, 17986564789705952593ull);
    vlSelf->__PVT__will_write_1 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6565345871816459926ull);
    VL_SCOPED_RAND_RESET_W(98, vlSelf->__PVT__out1_msg_reg, __VscopeHash, 5169445537384967357ull);
    vlSelf->__PVT__will_write_2 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5422800433389864155ull);
    VL_SCOPED_RAND_RESET_W(98, vlSelf->__PVT__out2_msg_reg, __VscopeHash, 18426009575718852139ull);
    vlSelf->__PVT__will_write_3 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4142536949964877498ull);
    VL_SCOPED_RAND_RESET_W(98, vlSelf->__PVT__out3_msg_reg, __VscopeHash, 17590883977631325723ull);
    vlSelf->__PVT__out0_valid_reg__DOT__state = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3599503515460236611ull);
    vlSelf->__PVT__out1_valid_reg__DOT__state = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7087440448375990745ull);
    vlSelf->__PVT__out2_valid_reg__DOT__state = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13317179179747457449ull);
    vlSelf->__PVT__out3_valid_reg__DOT__state = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6494359942246635194ull);
}
