# Logic Synthesis

**circt-synth** \[_options_] \[_filename_]

A logic synthesis tool converting designs expressed in the 
[comb](../Dialects/Comb/_index.md) and [HW](../Dialects/HW/_index.md) dialects.
There are no additional build requirements and can build directly 
`ninja circt-synth`. The tool currently only supports combinatorial logic but we
are actively working to extend support to sequential designs (get involved).

By default the design is lowered to an and-inverter graph (AIG) represented
using the [synth](../Dialects/Synth/_index.md) dialect using optimisations that
target general cell libraries. A `--synthesis-strategy` option can configure 
area vs delay optimisation. The tool uses a longest path analysis to guide
design decisions and can output the critical path to a file via the 
`--output-longest-path=<filename>` option. 

### Example

```mlir
// sum.mlir
module {
  hw.module @sum(in %a : i4, in %b : i4, out sum : i4) {
    %0 = comb.add %a, %b : i4
    hw.output %0 : i4
  }
}
```

Writing `circt-synth sum.mlir` the addition is lowered to the
[synth](../Dialects/Synth/_index.md) dialect that expresses the design using
only and-inverter operations and `concat`/`extract` operations:
```mlir
module {
  hw.module private @sum(in %a : i4, in %b : i4, out sum : i4) {
    %0 = comb.extract %a from 0 : (i4) -> i1
    %1 = comb.extract %a from 1 : (i4) -> i1
    %2 = comb.extract %a from 2 : (i4) -> i1
    %3 = comb.extract %a from 3 : (i4) -> i1
    %4 = comb.extract %b from 0 : (i4) -> i1
    %5 = comb.extract %b from 1 : (i4) -> i1
    %6 = comb.extract %b from 2 : (i4) -> i1
    %7 = comb.extract %b from 3 : (i4) -> i1
    %8 = synth.aig.and_inv not %0, not %4 : i1
    %9 = synth.aig.and_inv %0, %4 : i1
    %10 = synth.aig.and_inv not %8, not %9 : i1
    %11 = synth.aig.and_inv not %1, not %5 : i1
    ...
    %31 = comb.concat %30, %23, %16, %10 : i1, i1, i1, i1
    hw.output %31 : i4
  }
}
```