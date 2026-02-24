// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VESI_Cosim_Top.h for the primary calling header

#ifndef VERILATED_VESI_COSIM_TOP_READMEM_WIDTH512_H_
#define VERILATED_VESI_COSIM_TOP_READMEM_WIDTH512_H_  // guard

#include "verilated.h"


class VESI_Cosim_Top__Syms;

class alignas(VL_CACHE_LINE_BYTES) VESI_Cosim_Top_ReadMem_width512 final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    // Anonymous structures to workaround compiler member-count bugs
    struct {
        VL_IN8(clk,0,0);
        VL_IN8(rst,0,0);
        VL_IN8(cmd_512_cmd_valid,0,0);
        VL_IN8(addrCmdCycles_get_valid,0,0);
        VL_IN8(addrCmdIssued_get_valid,0,0);
        VL_IN8(addrCmdResponses_get_valid,0,0);
        VL_IN8(host_resp_valid,0,0);
        VL_IN8(lastReadLSB_get_valid,0,0);
        VL_IN8(cmd_512_data_ready,0,0);
        VL_IN8(addrCmdCycles_data_ready,0,0);
        VL_IN8(addrCmdIssued_data_ready,0,0);
        VL_IN8(addrCmdResponses_data_ready,0,0);
        VL_IN8(host_req_ready,0,0);
        VL_IN8(lastReadLSB_data_ready,0,0);
        VL_OUT8(cmd_512_cmd_ready,0,0);
        VL_OUT8(addrCmdCycles_get_ready,0,0);
        VL_OUT8(addrCmdIssued_get_ready,0,0);
        VL_OUT8(addrCmdResponses_get_ready,0,0);
        VL_OUT8(host_resp_ready,0,0);
        VL_OUT8(lastReadLSB_get_ready,0,0);
        VL_OUT8(cmd_512_data_valid,0,0);
        VL_OUT8(addrCmdCycles_data_valid,0,0);
        VL_OUT8(addrCmdIssued_data_valid,0,0);
        VL_OUT8(addrCmdResponses_data_valid,0,0);
        VL_OUT8(host_req_valid,0,0);
        VL_OUT8(lastReadLSB_data_valid,0,0);
        CData/*0:0*/ __PVT___pipelineStage_a_ready;
        CData/*0:0*/ __PVT__address_command__DOT__cmd_512_data_ready;
        CData/*0:0*/ __PVT__address_command__DOT__addrCmdCycles_get_ready;
        CData/*0:0*/ __PVT__address_command__DOT__addrCmdIssued_get_ready;
        CData/*0:0*/ __PVT__address_command__DOT__addrCmdResponses_get_ready;
        CData/*0:0*/ __PVT__address_command__DOT__command_go;
        CData/*0:0*/ __PVT__address_command__DOT__hostmem_cmd_address_valid;
        CData/*0:0*/ __PVT__address_command__DOT___pipelineStage_a_ready;
        CData/*0:0*/ __PVT__address_command__DOT___pipelineStage_a_ready_0;
        CData/*0:0*/ __PVT__address_command__DOT___pipelineStage_a_ready_1;
        CData/*0:0*/ __PVT__address_command__DOT___GEN;
        CData/*0:0*/ __PVT__address_command__DOT___GEN_1;
        CData/*0:0*/ __PVT__address_command__DOT__operation_active__DOT__state;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage__DOT__x_valid_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage__DOT__x_ready_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage__DOT__xmit;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage__DOT__l_valid;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_0__DOT__x_valid_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_0__DOT__x_ready_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_0__DOT__xmit;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_0__DOT__l_valid;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_0__DOT__a_rcv;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_1__DOT__x_valid_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_1__DOT__x_ready_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_1__DOT__xmit;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_1__DOT__l_valid;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_2__DOT__x_valid_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_2__DOT__x_ready_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_2__DOT__xmit;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_2__DOT__l_valid;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_2__DOT__a_rcv;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_3__DOT__x_valid_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_3__DOT__x_ready_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_3__DOT__xmit;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_3__DOT__l_valid;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_4__DOT__x_valid_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_4__DOT__x_ready_reg;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_4__DOT__xmit;
    };
    struct {
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_4__DOT__l_valid;
        CData/*0:0*/ __PVT__address_command__DOT__pipelineStage_4__DOT__a_rcv;
        CData/*0:0*/ __PVT__pipelineStage__DOT__x_valid_reg;
        CData/*0:0*/ __PVT__pipelineStage__DOT__x_ready_reg;
        CData/*0:0*/ __PVT__pipelineStage__DOT__xmit;
        CData/*0:0*/ __PVT__pipelineStage__DOT__l_valid;
        CData/*0:0*/ __PVT__pipelineStage_0__DOT__x_valid_reg;
        CData/*0:0*/ __PVT__pipelineStage_0__DOT__x_ready_reg;
        CData/*0:0*/ __PVT__pipelineStage_0__DOT__xmit;
        CData/*0:0*/ __PVT__pipelineStage_0__DOT__l_valid;
        CData/*0:0*/ __PVT__pipelineStage_0__DOT__a_rcv;
        VL_INW(cmd_512_cmd,96,0,4);
        VL_INW(host_resp,519,0,17);
        VL_OUT64(cmd_512_data,63,0);
        VL_OUT64(addrCmdCycles_data,63,0);
        VL_OUT64(addrCmdIssued_data,63,0);
        VL_OUT64(addrCmdResponses_data,63,0);
        VL_OUTW(host_req,71,0,3);
        VL_OUT64(lastReadLSB_data,63,0);
        QData/*63:0*/ __PVT__last_read_lsb___05Freg1;
        QData/*63:0*/ __PVT__address_command__DOT__start_addr;
        QData/*63:0*/ __PVT__address_command__DOT__flits_total;
        QData/*63:0*/ __PVT__address_command__DOT__addr_cmd_cycles___05Freg1;
        QData/*63:0*/ __PVT__address_command__DOT__addr_cmd_responses_cnt__DOT___GEN;
        QData/*63:0*/ __PVT__address_command__DOT__addr_cmd_cycle_counter__DOT___GEN;
        QData/*63:0*/ __PVT__address_command__DOT__Counter__DOT___GEN;
        QData/*63:0*/ __PVT__address_command__DOT__pipelineStage__DOT__x_reg;
        QData/*63:0*/ __PVT__address_command__DOT__pipelineStage__DOT__l;
        QData/*63:0*/ __PVT__address_command__DOT__pipelineStage_1__DOT__x_reg;
        QData/*63:0*/ __PVT__address_command__DOT__pipelineStage_1__DOT__l;
        QData/*63:0*/ __PVT__address_command__DOT__pipelineStage_3__DOT__x_reg;
        QData/*63:0*/ __PVT__address_command__DOT__pipelineStage_3__DOT__l;
        QData/*63:0*/ __PVT__pipelineStage__DOT__x_reg;
        QData/*63:0*/ __PVT__pipelineStage__DOT__l;
    };

    // INTERNAL VARIABLES
    VESI_Cosim_Top__Syms* const vlSymsp;

    // CONSTRUCTORS
    VESI_Cosim_Top_ReadMem_width512(VESI_Cosim_Top__Syms* symsp, const char* v__name);
    ~VESI_Cosim_Top_ReadMem_width512();
    VL_UNCOPYABLE(VESI_Cosim_Top_ReadMem_width512);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
