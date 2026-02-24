// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VESI_COSIM_TOP__SYMS_H_
#define VERILATED_VESI_COSIM_TOP__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "VESI_Cosim_Top.h"

// INCLUDE MODULE CLASSES
#include "VESI_Cosim_Top___024root.h"
#include "VESI_Cosim_Top___024unit.h"
#include "VESI_Cosim_Top_Cosim_DpiPkg.h"

// DPI TYPES for DPI Export callbacks (Internal use)

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES) VESI_Cosim_Top__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    VESI_Cosim_Top* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    VESI_Cosim_Top___024root       TOP;
    VESI_Cosim_Top_Cosim_DpiPkg    TOP__Cosim_DpiPkg;

    // SCOPE NAMES
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_req_data;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_req_data__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_resp_data;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_resp_data__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_resp_data__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_read_resp_data__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_arg;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_arg__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_result;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_result__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_result__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_hostmem_write_result__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_arg;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_arg__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_arg__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_arg__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_result;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_result__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcosim_mmio_read_write_result__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcycle_counter__req_ep;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcycle_counter__req_ep__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcycle_counter__req_ep__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcycle_counter__req_ep__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcycle_counter__resp_ep;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcycle_counter__resp_ep__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top_____05Fcycle_counter__resp_ep__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__arrayFunc_arg;
    VerilatedScope __Vscope_ESI_Cosim_Top__arrayFunc_arg__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top__arrayFunc_arg__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__arrayFunc_arg__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__arrayFunc_result;
    VerilatedScope __Vscope_ESI_Cosim_Top__arrayFunc_result__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__arrayFunc_result__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__func1_arg;
    VerilatedScope __Vscope_ESI_Cosim_Top__func1_arg__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top__func1_arg__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__func1_arg__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__func1_result;
    VerilatedScope __Vscope_ESI_Cosim_Top__func1_result__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__func1_result__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_fromhw_send;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_fromhw_send__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_fromhw_send__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_tohw_recv;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_tohw_recv__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_tohw_recv__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_loopback_tohw_recv__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_recv_recv;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_recv_recv__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_recv_recv__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_recv_recv__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_send_send;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_send_send__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B05D_mysvc_send_send__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_fromhw_send;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_fromhw_send__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_fromhw_send__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_tohw_recv;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_tohw_recv__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_tohw_recv__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_loopback_tohw_recv__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_recv_recv;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_recv_recv__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_recv_recv__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_recv_recv__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_send_send;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_send_send__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__loopback_inst5B15D_mysvc_send_send__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__oddStructFunc_arg;
    VerilatedScope __Vscope_ESI_Cosim_Top__oddStructFunc_arg__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top__oddStructFunc_arg__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__oddStructFunc_arg__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__oddStructFunc_result;
    VerilatedScope __Vscope_ESI_Cosim_Top__oddStructFunc_result__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__oddStructFunc_result__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__structFunc_arg;
    VerilatedScope __Vscope_ESI_Cosim_Top__structFunc_arg__out_pipe;
    VerilatedScope __Vscope_ESI_Cosim_Top__structFunc_arg__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__structFunc_arg__unnamedblk2;
    VerilatedScope __Vscope_ESI_Cosim_Top__structFunc_result;
    VerilatedScope __Vscope_ESI_Cosim_Top__structFunc_result__unnamedblk1;
    VerilatedScope __Vscope_ESI_Cosim_Top__structFunc_result__unnamedblk2;

    // CONSTRUCTORS
    VESI_Cosim_Top__Syms(VerilatedContext* contextp, const char* namep, VESI_Cosim_Top* modelp);
    ~VESI_Cosim_Top__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
};

#endif  // guard
