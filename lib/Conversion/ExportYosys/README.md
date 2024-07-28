# Yosys Integration

Yosys integration is conditionally enabled by cmake flag.


## Build

CIRCT currently requires manual instation of 


## Tranlation between RTLIL

The abstraction level of RTLIL is generally equal to CIRCT core dialects.
At this point we don't support translation of procedures.


## Run Yosys passes on CIRCT IR

To run Yosys passes on CIRCT there are two passes.  
`export-yosys-parallel` and `export-yosys`.
### CIRCT as a parallel Yosys driver

Yosys has a globally context which is a not thread-safe so we cannot parallely run synthesizer with multhi threads. CIRCT provides a `export-yosys-parallel` pass that invokes yosys in child processes. `export-yosys-parallel` cannot be used for transformation that requires module hierarchly (e.g. inlining/flattening etc)


## CIRCT IR -> RTLIL
```
circt-tranlsate --export-rtlil
```

## RTLIL -> CIRCT IR

```
circt-tranlsate --import-rtlil
```
