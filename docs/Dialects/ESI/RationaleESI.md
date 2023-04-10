# System Construction: The Elastic Silicon Interconnect (ESI) dialect

At its core, system construction is "just" IP stitching -- gluing modules
together into coherent "system". In practice, however, that simple task solves
only one problem of many. ESI will provide:

- IP stitching
  - Type safety -- static safety errors
  - Elastic (latency insensitive) communication
  - "Gearboxing" large data buses down to smaller ones
  - Wire signaling protocol adaptation (i.e. FIFO vs Ready/Valid vs AXI streaming)
  - Automatic construction of clock domain crossings
  - Variable length types
  - Interconnect pipelining
- Multi-HDL support
  - Specification-based
- Board support packages
  - Services that implement abstract logical interfaces from board-specific communication
- Software (host) API generation
  - Automatic, typed generation in multiple languages
  - Same API regardless of communication method (e.g. real hardware vs cosim)
  - Host-side multi-process orchestration
- Separate modeling/specification from implementation
  - Simple specification (with optional additional specifications)
  - Performant implementations
- System co-simulation
  - RTL-level simulation
  - Emulation of ESI interconnect (CAM interop)

- Runtime discovery of services/IP
  - IP component meta-data (e.g. version, description)
  - Types for external communication
  - External communication methods (e.g. DMA engines, control offsets, etc)
- Runtime telemetry
  - Standardized collection methods
  - Standardized reporting methods
- Debugging
  - System-level debug (e.g. interconnect monitors)
  - HDL interface for language-specific debug
- IP/system configuration / readout
  - Standardized setup/tear down
- Clock and reset domains
  - IP assignment (to clk/rst domain)
  - Soft reset of IP
  - Standardized reset protocol
  - Shared resource (e.g. DRAM) / reset orchestration
- Multi-device orchestration
  - Host side orchestration
  - Device-device comms

## Status

ESI supports only basic features and those are not yet production ready.

## Publications

"Elastic Silicon Interconnects: Abstracting Communication in Accelerator
Design", John Demme (Microsoft).
[paper](https://capra.cs.cornell.edu/latte21/paper/8.pdf),
[talk](https://www.youtube.com/watch?v=gjOkGX2E7EY).
