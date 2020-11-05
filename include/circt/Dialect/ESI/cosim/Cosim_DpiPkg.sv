// Copyright (c) Microsoft Corporation. All rights reserved.
// =============================================================================
// Package: CosimCore_DpiPkg
//
// Authors:
// - John Demme (john.demme@microsoft.com)
// - Hari Angepat (hari.angepat@microsoft.com)
// - Andrew Lenharth (andrew.lenharth@microsoft.com)
//
// Description:
//   DPI-exposed funcs for cosimserver cosimulation unit-test
// =============================================================================

package Cosim_DpiPkg;

// --------------------- Cosim Socket Server ------------------------------

// Start cosimserver (spawns server for RTL-initiated work, listens for connections from new SW-clients)
import "DPI-C" sv2c_cosimserver_init = function int cosim_init();

// Teardown cosimserver (disconnects from primary server port, stops connections from active clients)
import "DPI-C" sv2c_cosimserver_fini = function void cosim_fini();

// --------------------- Endpoint Management ------------------------------

// Register simulated device endpoints
// - return 0 on success, non-zero on failure (duplicate EP registered)
import "DPI-C" sv2c_cosimserver_ep_register =
  function int cosim_ep_register(
      input int endpoint_id,
      input longint esi_type_id,
      input int type_size);

// --------------------- Endpoint Accessors ------------------------------

// Access a simulated device endpoint (only 1 active client supported per endpoint)
// - return 0 on success, negative on failure (unregistered EP)
// - tryget may return 0 but provide size_bytes=0
// - size_bytes can be used to indicate the actual number of valid bytes provided in the data[] array
// => ie) can allocate a fixed max-length data[4096] tmp buffer that is used for all DPI calls, while supporting variable length messages

//returns number of messages in queue.  negative for error.
// sets msg_size to size of first message.  Some implementations may not indicate more than one message in the queue despite multiple messages.
import "DPI-C" sv2c_cosimserver_ep_test =
  function int cosim_ep_test(
    input int unsigned endpoint_id,
    output int unsigned msg_size);

//returns if client is still alive.  negative for error
import "DPI-C" sv2c_cosimserver_conn_connected = 
   function int cosim_conn_connected(input int unsigned endpoint_id);

// Access a simulated device endpoint (multi-client supported per endpoint)
// - return 0 on success, negative on failure (unregistered EP)
import "DPI-C" sv2c_cosimserver_ep_tryput = 
  function int cosim_ep_tryput(
    input int unsigned endpoint_id,
    input byte unsigned data[],
    input int data_limit = -1
    );

// returns number of missing bytes (bytes beyond size_bytes) on success and sets size = number of bytes in message
// Sufficient buffers should always return 0 on success
// on failure, return negative.  
//If no message, return 0 with size == 0
import "DPI-C" sv2c_cosimserver_ep_tryget = 
  function int cosim_ep_tryget(
    input  int unsigned endpoint_id,
    inout byte unsigned data[], // This should be 'output', but Verilator doesn't seem to have a way to do this
    inout  int unsigned size_bytes
    );

endpackage // CoSim_Dpi
