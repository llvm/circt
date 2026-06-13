// RUN: circt-opt --lower-esi-ports %s --verify-diagnostics --split-input-file

// Channels nested more than one array deep cannot be lowered.
// expected-error @+2 {{cannot lower port containing channels nested inside an aggregate other than a single array of channels}}
// expected-note @+1 {{}}
hw.module @nestedArrayOfChannels(in %in: !hw.array<2 x !hw.array<3 x !esi.channel<i8>>>) {
  hw.output
}

// -----

// Channels nested inside a struct (inside an array) cannot be lowered.
// expected-error @+2 {{cannot lower port containing channels nested inside an aggregate other than a single array of channels}}
// expected-note @+1 {{}}
hw.module @arrayOfStructOfChannels(in %in: !hw.array<2 x !hw.struct<a: !esi.channel<i8>>>) {
  hw.output
}
