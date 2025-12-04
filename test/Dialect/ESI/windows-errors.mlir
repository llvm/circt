// RUN: circt-opt %s --split-input-file --verify-diagnostics

// Test: only windows into structs are supported
// expected-error @+1 {{only windows into structs are currently supported}}
!InvalidWindow = !esi.window<"invalid", i32, [<"Frame", [<"field">]>]>

// -----

// Test: invalid field name
!Struct = !hw.struct<field1: i32, field2: i16>
// expected-error @+1 {{invalid field name: "nonexistent"}}
!InvalidFieldWindow = !esi.window<"invalid", !Struct, [<"Frame", [<"nonexistent">]>]>

// -----

// Test: field already consumed by a previous frame
!Struct2 = !hw.struct<field1: i32, field2: i16>
// expected-error @+1 {{field '"field1"' already consumed by a previous frame}}
!DuplicateFieldWindow = !esi.window<"dup", !Struct2, [
  <"Frame1", [<"field1">]>,
  <"Frame2", [<"field1">]>
]>

// -----

// Test: cannot specify both numItems and countWidth
!ListStruct = !hw.struct<data: !esi.list<i32>>
!BothNumItemsAndCountWidth = !esi.window<"invalid", !ListStruct, [
  // expected-error @below {{cannot specify both numItems and countWidth for field 'data'}}
  // expected-error @below {{failed to parse WindowFrameType parameter 'members'}}
  // expected-error @below {{failed to parse ESIWindowType parameter 'frames'}}
  <"Frame", [<"data", 4 countWidth 16>]>
]>

// -----

// Test: numItems larger than array size
!ArrayStruct = !hw.struct<arr: !hw.array<4xi32>>
// expected-error @+1 {{num items is larger than array size in field "arr"}}
!NumItemsTooLarge = !esi.window<"invalid", !ArrayStruct, [
  <"Frame", [<"arr", 8>]>
]>

// -----

// Test: numItems only allowed on array or list fields
!ScalarStruct = !hw.struct<scalar: i32>
// expected-error @+1 {{specification of num items only allowed on array or list fields (in "scalar")}}
!NumItemsOnScalar = !esi.window<"invalid", !ScalarStruct, [
  <"Frame", [<"scalar", 4>]>
]>

// -----

// Test: countWidth only allowed on list fields
!ArrayStruct2 = !hw.struct<arr: !hw.array<4xi32>>
// expected-error @+1 {{bulk transfer (countWidth) only allowed on list fields (in "arr")}}
!CountWidthOnArray = !esi.window<"invalid", !ArrayStruct2, [
  <"Frame", [<"arr" countWidth 16>]>
]>

// -----

// Test: countWidth only allowed on list fields (scalar case)
!ScalarStruct2 = !hw.struct<scalar: i32>
// expected-error @+1 {{bulk transfer (countWidth) only allowed on list fields (in "scalar")}}
!CountWidthOnScalar = !esi.window<"invalid", !ScalarStruct2, [
  <"Frame", [<"scalar" countWidth 16>]>
]>

// -----

// Test: cannot have two array fields with numItems in same frame
!TwoArraysStruct = !hw.struct<arr1: !hw.array<8xi32>, arr2: !hw.array<8xi16>>
// expected-error @+1 {{cannot have two array or list fields with num items (in "arr2")}}
!TwoArraysWithNumItems = !esi.window<"invalid", !TwoArraysStruct, [
  <"Frame", [<"arr1", 4>, <"arr2", 4>]>
]>

// -----

// Test: non-bulk-transfer field cannot be reused with bulk transfer header
!NonBulkStruct = !hw.struct<dst: i32, payload: !esi.list<i32>>
// expected-error @+1 {{field '"payload"' already consumed by a previous frame}}
!NonBulkReuse = !esi.window<"invalid", !NonBulkStruct, [
  <"Frame1", [<"payload">]>,
  <"Frame2", [<"payload" countWidth 16>]>
]>

// -----

// Test: bulk transfer field cannot have countWidth specified twice
!DuplicateCountWidthStruct = !hw.struct<dst: i32, payload: !esi.list<i32>>
// expected-error @+1 {{field '"payload"' already has countWidth specified}}
!DuplicateCountWidth = !esi.window<"invalid", !DuplicateCountWidthStruct, [
  <"HeaderFrame1", [<"dst">, <"payload" countWidth 16>]>,
  <"HeaderFrame2", [<"payload" countWidth 8>]>
]>

// -----

// Test: bulk transfer data frame cannot appear twice
!DuplicateDataFrameStruct = !hw.struct<dst: i32, payload: !esi.list<i32>>
// expected-error @+1 {{field '"payload"' already consumed by a previous frame}}
!DuplicateDataFrame = !esi.window<"invalid", !DuplicateDataFrameStruct, [
  <"HeaderFrame", [<"dst">, <"payload" countWidth 16>]>,
  <"DataFrame1", [<"payload", 4>]>,
  <"DataFrame2", [<"payload", 2>]>
]>
