//===- LLCosim_DpiPkg.sv - ESI llcosim DPI declarations -----*- verilog -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Package: LLCosim_DpiPkg
//
// DPI-exposed funcs for low level cosimulation.
//
//===----------------------------------------------------------------------===//

package LLCosim_DpiPkg;

// --------------------- Cosim RPC Server --------------------------------------

// Start cosimserver (spawns server for HW-initiated work, listens for
// connections from new SW-clients).
import "DPI-C" sv2cLLCosimserverInit = function int llcosim_init();

// Teardown cosimserver (disconnects from primary server port, stops connections
// from active clients).
import "DPI-C" sv2cLLCosimserverFinish = function void llcosim_finish();

endpackage // LLCosim_DpiPkg
