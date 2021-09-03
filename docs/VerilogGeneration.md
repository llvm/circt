# Verilog and SystemVerilog Generation

Verilog and SystemVerilog are critical components of the hardware design tool
ecosystem, but generating output that is correct and acceptable by a wide range
of tools is a challenge, and generating "good looking" output even more so.
This document describes CIRCT's approach and support for generating Verilog,
some of the features and capabilities provided, and information about the
internal layering of the related subsystems.

## Why is this hard?

One of the goals of CIRCT is to insulate "front end" authors from the details of
Verilog generation.  We would like to see innovation at the authoring level, and
the problems in that space are quite different than the challenges of creating
syntactically correct Verilog.

Further, the Verilog/SystemVerilog languages were primarily designed to be a
human-authored programming language, and have evolved over the years with many
new and exciting features.  At the same time, the industry is full of critical
EDA tools - but these have mixed support for different language features.  Open
source tools in particular have [mixed
support](https://chipsalliance.github.io/sv-tests-results/) for new features.
Different emission styles also impact [simulator performance](http://www.sunburst-design.com/papers/CummingsDVCon2019_Yikes_SV_Coding_rev1_0.pdf)
and have many other considerations.  We would like clients of CIRCT to be insulated from this complexity where possible.

Beyond the capabilities of different tools, in many cases the output of CIRCT
is run through various "linters" that look for antipatterns or possible bugs in
the output.  While it is difficult to work with arbitrary 3rd party 

Finally, our goal is for the generated Verilog to be as readable and polished as
possible - some users of CIRCT generate IP that is sold to customers, and the
quality of the generated Verilog directly reflects on the quality of the
corresponding products.  This means that small details, including indentation
and use of the correct idioms is important.

## Controlling output style with `LoweringOptions`

The primary interface to control the output style from a CIRCT-based tool is
through the [`circt::LoweringOptions`](https://github.com/llvm/circt/blob/main/include/circt/Support/LoweringOptions.h)
structure.  It contains a number of properties (e.g. `emittedLineLength` or
`disallowLocalVariables`) that affect lowering and emission of Verilog -- in
this case, what length of lines the emitter should aim for (e.g. 80 columns
wide, 120 wide, etc), and whether the emitter is allowed to use `automatic
logic` declarations in nested blocks or not.

The defaults in `LoweringOptions` are set up to generate aethetically pleasing
output, and to use the modern features of SystemVerilog where possible.  Client
tools and frontends can change these, e.g. if they need to generate standard
Verilog for older tools.

This struct provides a convenient way to work with lowering options for C++
clients, but the actual truth is encoded into the IR as an attribute - the
`circt.loweringOptions` string attribute on the top level `builtin.module`
declaration.  Any frontend can set these options manually that way.

Command line tools also generally provide a `--lowering-options=` flag that
allows end-users to override the defaults or the front-end provided features.
If you're using `firtool` for example, you can pass
`--lowering-options=emittedLineLength=200` to change the line length.  This can
be useful for experimentation, or when a frontend doesn't have other ways to
control the output.

The current set of "tool capability" Lowering Options is:

 * `useAlwaysFF` (default=`false`).  If true, emits `sv.alwaysff` as
    Verilog `always_ff` statements.  Otherwise, print them as `always` statements.
 * `useAlwaysComb` (default=`false`).  If true, emits `sv.alwayscomb` as Verilog
   `always_comb` statements.  Otherwise, print them as `always @(*)`.
 * `allowExprInEventControl` (default=`false`).   If true, expressions are
   allowed in the sensitivity list of `always` statements, otherwise they are
   forced to be simple wires. Some EDA tools rely on these being simple wires.
 * `disallowPackedArrays` (default=`false`).  If true, eliminate packed arrays
   for tools that don't support them (e.g. Yosys).
 * `disallowLocalVariables` (default=`false`).  If true, do not emit
   SystemVerilog locally scoped "automatic" or logic declarations - emit top
   level wire and reg's instead.
 * `enforceVerifLabels` (default=`false`).  If true, verification statements
   like `assert`, `assume`, and `cover` will always be emitted with a label. If
   the statement has no label in the IR, a generic one will be created. Some EDA
   tools require verification statements to be labeled.
  
The current set of "style" Lowering Options is:

 * `emittedLineLength` (default=`90`).  This is the target width of lines in an
   emitted Verilog source file in columns.

## Adding new Lowering Options

Use active verbs.  Default to pretty and new.  Keep consistent naming style with
other options. Keep the dox above up to date.


## Verilog Exporter Internals

Pass pipeline.  Separation of responsibility between prettify verilog

Verilog Emitter Prepass

Expectations of ExportVerilog itself
 - Cyclic references get wires.
 - things are sunk into their blocks.

### PrettifyVerilog


### ExportVerilog