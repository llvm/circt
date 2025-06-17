# LTL Dialect

This dialect provides operations and types to model [Linear Temporal Logic](https://en.wikipedia.org/wiki/Linear_temporal_logic), sequences, and properties, which are useful for hardware verification.

[TOC]


## Rationale

The main goal of the `ltl` dialect is to capture the core formalism underpinning SystemVerilog Assertions (SVAs), the de facto standard for describing temporal logic sequences and properties in hardware verification. (See IEEE 1800-2017 section 16 "Assertions".) We expressly try *not* to model this dialect like an AST for SVAs, but instead try to strip away all the syntactic sugar and Verilog quirks, and distill out the core foundation as an IR. Within the CIRCT project, this dialect intends to enable emission of rich temporal assertions as part of the Verilog output, but also provide a foundation for formal tools built ontop of CIRCT.


### Sequences and Properties

The core building blocks for modeling temporal logic in the `ltl` dialect are *sequences* and *properties*. In a nutshell, sequences behave like regular expressions over time, whereas properties provide the quantifiers to express that sequences must be true under certain conditions.

**Sequences** describe boolean expressions at different points in time. They can be easily verified by a finite state automaton, similar to how regular expressions and languages have an equivalent automaton that recognizes the language. For example:

- The boolean `a` is a sequence. It holds if `a` is true in cycle 0 (the current cycle).
- The boolean expression `a & b` is also a sequence. It holds if `a & b` is true in cycle 0.
- `##1 a` checks that `a` is true in cycle 1 (the next cycle).
- `##[1:4] a` checks that `a` is true anywhere in cycle 1, 2, 3, or 4.
- `a ##1 b` checks that `a` holds in cycle 0 and `b` holds in cycle 1.
- `##1 (a ##1 b)` checks that `a` holds in cycle 1 and `b` holds in cycle 2.
- `(a ##1 b) ##5 (c ##1 d)` checks that the sequence `(a ##1 b)` holds and is followed by the sequence `(c ##1 d)` 5 or 6 cycles later. Concretely, this checks that `a` holds in cycle 0, `b` holds in cycle 1, `c` holds in cycle 6 (5 cycles after the first sequence ended in cycle 1), and `d` holds in cycle 7.

**Properties** describe concrete, testable propositions or claims built from sequences. While sequences can observe and match a certain behavior in a circuit at a specific point in time, properties allow you to express that these sequences hold in every cycle, or hold at some future point in time, or that one sequence is always followed by another. For example:

- `always s` checks that the sequence `s` holds in every cycle. This is often referred to as the **G** (or "globally") operator in LTL.
- `eventually s` checks that the sequence `s` will hold at some cycle now or in the future. This is often referred to as the **F** (or "finally") operator in LTL.
- `p until q` checks that the property `p` holds in every cycle before the first cycle `q` holds. This is often referred to as the **U** (or "until") operator in LTL.
- `s implies t` checks that whenever the sequence `s` is observed, it is immediately followed by sequence `t`.

Traditional definitions of the LTL formalism do not make a distinction between sequences and properties. Most of their operators fall into the property category, for example, quantifiers like *globally*, *finally*, *release*, and *until*. The set of sequence operators is usually very small, since it is not necessary for academic treatment, consisting only of the *next* operator. The `ltl` dialect provides a richer set of operations to model sequences.


## Representing SVAs


### Sequence Concatenation and Cycle Delay

The primary building block for sequences in SVAs is the *concatenation* expression. Concatenation is always associated with a cycle delay, which indicates how many cycles pass between the end of the LHS sequence and the start of the RHS sequence. One, two, or more sequences can be concatenated at once, and the overall concatenation can have an initial cycle delay. For example:

```
a ##1 b ##1 c      // 1 cycle delay between a, b, and c
##2 a ##1 b ##1 c  // same, plus 2 cycles of initial delay before a
```

In the simplest form, a cycle delay can appear as a prefix of another sequence, e.g., `##1 a`. This is essentially a concatenation with only one sequence, `a`, and an initial cycle delay of the concatenation of `1`. The prefix delays map to the LTL dialect as follows:

- `##N seq`. **Fixed delay.** Sequence `seq` has to match exactly `N` cycles in the future. Equivalent to `ltl.delay %seq, N, 0`.
- `##[N:M] seq`. **Bounded range delay.** Sequence `seq` has to match anywhere between `N` and `M` cycles in the future, inclusive. Equivalent to `ltl.delay %seq, N, (M-N)`
- `##[N:$] seq`. **Unbounded range delay.** Sequence `seq` has to match anywhere at or beyond `N` cycles in the future, after a finite amount of cycles. Equivalent to `ltl.delay %seq, N`.
- `##[*] seq`. Shorthand for `##[0:$]`. Equivalent to `ltl.delay %seq, 0`.
- `##[+] seq`. Shorthand for `##[1:$]`. Equivalent to `ltl.delay %seq, 1`.

Concatenation of two sequences always involves a cycle delay specification in between them, e.g., `a ##1 b` where sequence `b` starts in the cycle after `a` ends. Zero-cycle delays can be specified, e.g., `a ##0 b` where `b` starts in the same cycle as `a` ends. If `a` and `b` are booleans, `a ##0 b` is equivalent to `a && b`.

The dialect separates concatenation and cycle delay into two orthogonal operations, `ltl.concat` and `ltl.delay`, respectively. The former models concatenation as `a ##0 b`, and the latter models delay as a prefix `##1 c`. The SVA concatenations with their infix delays map to the LTL dialect as follows:

- `seqA ##N seqB`. **Binary concatenation.** Sequence `seqB` follows `N` cycles after `seqA`. This can be represented as `seqA ##0 (##N seqB)`, which is equivalent to
  ```
  %0 = ltl.delay %seqB, N, 0
  ltl.concat %seqA, %0
  ```

- `seqA ##N seqB ##M seqC`. **Variadic concatenation.** Sequence `seqC` follows `M` cycles after `seqB`, which itself follows `N` cycles after `seqA`. This can be represented as `seqA ##0 (##N seqB) ##0 (##M seqC)`, which is equivalent to
  ```
  %0 = ltl.delay %seqB, N, 0
  %1 = ltl.delay %seqC, M, 0
  ltl.concat %seqA, %0, %1
  ```
  Since concatenation is associative, this is also equivalent to `seqA ##N (seqB ##M seqC)`:
  ```
  %0 = ltl.delay %seqC, M, 0
  %1 = ltl.concat %seqB, %0
  %2 = ltl.delay %1, N, 0
  ltl.concat %seqA, %2
  ```
  And also `(seqA ##N seqB) ##M seqC`:
  ```
  %0 = ltl.delay %seqB, N, 0
  %1 = ltl.concat %seqA, %0
  %2 = ltl.delay %seqC, M, 0
  ltl.concat %1, %2
  ```

- `##N seqA ##M seqB`. **Initial delay.** Sequence `seqB` follows `M` cycles afer `seqA`, which itself starts `N` cycles in the future. This is equivalent to a delay on `seqA` within the concatenation:
  ```
  %0 = ltl.delay %seqA, N, 0
  %1 = ltl.delay %seqB, M, 0
  ltl.concat %0, %1
  ```
  Alternatively, the delay can also be placed on the entire concatenation:
  ```
  %0 = ltl.delay %seqB, M, 0
  %1 = ltl.concat %seqA, %0
  ltl.delay %1, N, 0
  ```

- Only the fixed delay `##N` is shown here for simplicity, but the examples extend to the other delay flavors `##[N:M]`, `##[N:$]`, `##[*]`, and `##[+]`.


### Implication

```
seq |-> prop
seq |=> prop
```

The overlapping `|->` and non-overlapping `|=>` implication operators of SVA, which only check a property after a precondition sequence matches, map to the `ltl.implication` operation. When the sequence matches in the overlapping case `|->`, the property check starts at the same time the matched sequence ended. In the non-overlapping case `|=>`, the property check starts *at the clock tick after the* end of the matched sequence, unless the matched sequence was empty, in which special rules apply. (See IEEE 1800-2017 section 16.12.7 "Implication".) The non-overlapping operator can be expressed in terms of the overlapping operator:

```
seq |=> prop
```
is equivalent to
```
(seq ##1 true) |-> prop
```

The `ltl.implication` op implements the overlapping case `|->`, such that the two SVA operator flavors map to the `ltl` dialect as follows:

- `seq |-> prop`. **Overlapping implication.** Equivalent to `ltl.implication %seq, %prop`.
- `seq |=> prop`. **Non-overlapping implication.** Equivalent to
  ```
  %true = hw.constant true
  %0 = ltl.delay %true, 1, 0
  %1 = ltl.concat %seq, %0
  ltl.implication %1, %prop
  ```

An important benefit of only modeling the overlapping `|->` implication operator is that it does not interact with a clock. The end point of the left-hand sequence is the starting point of the right-hand sequence. There is no notion of delay between the end of the left and the start of the right sequence. Compare this to the `|=>` operator in SVA, which implies that the right-hand sequence happens at "strictly the next clock tick", which requires the operator to have a notion of time and clocking. As described above, it is still possible to model this using an explicit `ltl.delay` op, which already has an established interaction with a clock.


### Repetition

Consecutive repetition repeats the sequence by a number of times. For example, `s[*3]` repeats the sequence `s` three times, which is equivalent to `s ##1 s ##1 s`. This also applies when the sequence `s` matches different traces with different lengths. For example `(##[0:3] a)[*2]` is equivalent to the disjunction of all the combinations such as `a ##1 a`, `a ##1 (##3 a)`, `(##3 a) ##1 (##2 a)`. However, the repetition with unbounded range cannot be expanded to the concatenations as it produces an infinite formula.

The definition of `ltl.repeat` is similar to that of `ltl.delay`. The mapping from SVA's consecutive repetition to the LTL dialect is as follows:

- `seq[*N]`. **Fixed repeat.** Repeats `N` times. Equivalent to `ltl.repeat %seq, N, 0`.
- `seq[*N:M]`. **Bounded range repeat.** Repeats `N` to `M` times. Equivalent to `ltl.repeat %seq, N, (M-N)`.
- `seq[*N:$]`. **Unbounded range repeat.** Repeats `N` to an indefinite finite number of times. Equivalent to `ltl.repeat %seq, N`.
- `seq[*]`. Shorthand for `seq[*0:$]`. Equivalent to `ltl.repeat %seq, 0`.
- `seq[+]`. Shorthand for `seq[*1:$]`. Equivalent to `ltl.repeat %seq, 1`.
  
#### Non-Consecutive Repetition  

Non-consecutive repetition checks that a sequence holds a certain number of times within an arbitrary repetition of the sequence. There are two ways of expressing non-consecutive repetition, either by including the last iteration in the count or not. If the last iteration is included, then this is called a "go-to" style non-consecutive repetition and can be defined using the `ltl.goto_repeat <input>, <N>, <window>` operation, e.g. `a !b b b !b !b b c` is a valid observation of `ltl.goto_repeat %b, 1, 2`, but `a !b b b !b !b b !b !b c` is not. If we omit the constraint of having the last iteration hold, then this is simply called a non-consecutive repetition, and can be defined using the `ltl.non_consecutive_repeat <input, <N>, <window>` operation, e.g. both `a !b b b !b !b b c` and `a !b b b !b !b b !b !b c` are valid observations of `ltl.non_consecutive_repeat %b, 1, 2`. The SVA mapping of these operations is as follows:

- `seq[->n:m]`: **Go-To Style Repetition**, equivalent to `ltl.goto_repeat %seq, n, (m-n)`.  
- `seq[=n:m]` : **Non-Consecutive Repetition** equivalent to `ltl.non_consecutive_repeat %seq, n, (m-n)`.  


### Clocking

Sequence and property expressions in SVAs can specify a clock with respect to which all cycle delays are expressed. (See IEEE 1800-2017 section 16.16 "Clock resolution".) These map to the `ltl.clock` operation.

- `@(posedge clk) seqOrProp`. **Trigger on low-to-high clock edge.** Equivalent to `ltl.clock %seqOrProp, posedge %clk`.
- `@(negedge clk) seqOrProp`. **Trigger on high-to-low clock edge.** Equivalent to `ltl.clock %seqOrProp, negedge %clk`.
- `@(edge clk) seqOrProp`. **Trigger on any clock edge.** Equivalent to `ltl.clock %seqOrProp, edge %clk`.


### Disable Iff

Properties in SVA can have a disable condition attached, which allows for preemptive resets to be expressed. If the disable condition is true at any time during the evaluation of a property, the property is considered disabled. (See IEEE 1800-2017 end of section 16.12 "Declaring properties".) This maps to the `ltl.disable` operation.

- `disable iff (expr) prop`. **Disable condition.** Equivalent to `ltl.disable %prop if %expr`.

Note that SVAs only allow for entire properties to be disabled, at the point at which they are passed to an assert, assume, or cover statement. It is explicitly forbidden to define a property with a `disable iff` clause and then using it within another property. For example, the following is forbidden:
```
property p0; disable iff (cond) a |-> b; endproperty
property p1; eventually p0; endproperty
```
In this example, `p1` refers to property `p0`, which is illegal in SVA since `p0` itself defines a disable condition.

In contrast, the LTL dialect explicitly allows for properties to be disabled at arbitrary points, and disabled properties to be used in other properties. Since a disabled nested property also disables the parent property, the IR can always be rewritten into a form where there is only one `disable iff` condition at the root of a property expression.  
  
## Mapping SVA to CIRCT dialects  
  
Knowing how to map SVA constructs to CIRCT is important to allow these to expressed correctly in any front-end. Here you will find a non-exhaustive list of conversions from SVA to CIRCT dialects.  

- **properties**: `!ltl.property`.    
- **sequences**: `!ltl.sequence`.  
  
- **`disable iff (cond)`**:  `ltl.disable %prop if %cond`
- **local variables**: Not currently possible to encode.  

- **`$rose(a)`**: 
```mlir
%1 = ltl.compreg %a, %clock : i1
%rose = comb.icmp bin ult %1, %a : i1  
```  

- **`$fell(a)`**: 
```mlir
%1 = ltl.compreg %a, %clock : i1
%fell = comb.icmp bin ugt %a, %1 : i1
```    

- **`$stable(a)`**:
```mlir
%1 = ltl.compreg %a, %clock : i1
%rose = comb.icmp bin eq %a, %1 : i1
``` 

- **`$past(a, n)`**: 
```mlir
%zero = hw.constant 0 : i1
%true = hw.constant 1 : i1
%1 = seq.shiftreg n, %a, %clk, %true, powerOn %zero : i1
``` 

> The following functions are not yet supported by CIRCT:  
> - **`$onehot(a)`**  
> - **`$onehot0(a)`**  
> - **`$isunknown(a)`**  
> - **`$countones(a)`**     
  
  
- **`a ##n b`**:   
```mlir
%a_n = ltl.delay %a, n, 0 : i1
%anb = ltl.concat %a_n, %b : !ltl.sequence 
```

- **`a ##[n:m] b`**:
```mlir
%a_n = ltl.delay %a, n, (m-n) : i1
%anb = ltl.concat %a_n, %b : !ltl.sequence   
```

- **`s [*n]`**:  
```mlir
%repsn = ltl.repeat %s, n, 0 : !ltl.sequence  
```

- **`s [*n:m]`**: 
```mlir
%repsnm = ltl.repeat %s, n, (m-n) : !ltl.sequence  
```

- **`s[*n:$]`**:   
```mlir
%repsninf = ltl.repeat %s, n : !ltl.sequence  
```

- **`s[->n:m]`**:   
```mlir
%1 = ltl.goto_repeat %s, n, (m-n) : !ltl.sequence
```  

- **`s[=n:m]`**:   
```mlir
%1 = ltl.non_consecutive_repeat %s, n, (m-n) : !ltl.sequence  
```  

- **`s1 ##[+] s2`**:   
```mlir
%ds1 = ltl.delay %s1, 1
%s1s2 = ltl.concat %ds1, %s2 : !ltl.sequence  
```  

- **`s1 ##[*] s2`**:   
```mlir
%ds1 = ltl.delay %s1, 0
%s1s2 = ltl.concat %ds1, %s2 : !ltl.sequence  
```  

- **`s1 and s2`**:   
```mlir
ltl.and %s1, %s2 : !ltl.sequence  
```

- **`s1 intersect s2`**:   
```mlir
ltl.intersect %s1, %s2 : !ltl.sequence  
```

- **`s1 or s2`**:   
```mlir
ltl.or %s1, %s2 : !ltl.sequence  
```

- **`not s`**:   
```mlir
ltl.not %s1 : !ltl.sequence   
```

- **`first_match(s)`**: Not possible to encode yet.

- **`expr throughout s`**:   
```mlir
%repexpr = ltl.repeat %expr, 0 : !ltl.sequence
%res = ltl.intersect %repexpr, %s : !ltl.sequence  
```

- **`s1 within s2`**:     
```mlir
%c1 = hw.constant 1 : i1
%rep1 = ltl.repeat %c1, 0 : !ltl.sequence
%drep1 = ltl.delay %rep1, 1, 0 : !ltl.sequence
%ds1 = ltl.delay %s1, 1, 0 : !ltl.sequence
%evs1 = ltl.concat %drep1, %ds1, %c1 : !ltl.sequence
%res = ltl.intersect %evs1, %s2 : !ltl.sequence  
```

- **`s |-> p`**:   
```mlir
%1 = ltl.implication %s, %p : !ltl.property
```  

- **`s |=> p`**:  
```mlir
%c1 = hw.constant 1 : i1
%ds = ltl.delay %s, 1, 0 : i1
%antecedent = ltl.concat %ds, %c1 : !ltl.sequence
%impl = ltl.implication %antecedent, %p : !ltl.property  
```

- **`p1 implies p2`**:   
```mlir
%np1 = ltl.not %p1 : !ltl:property
%impl = ltl.or %np1, %p2 : !ltl.property  
```

- **`p1 iff p2`**: equivalent to `(not (p1 or p2)) or (p1 and p2)`  
```mlir
%p1orp2 = ltl.or %p1, %p2 : !ltl.property
%lhs = ltl.not %p1orp2 : !ltl.property
%rhs = ltl.and %p1, %p2 : !ltl.property
%iff = ltl.or %lhs, %rhs : !ltl.property  
```

- **`s #-# p`**:   
```mlir
%np = ltl.not %p : !ltl.property
%impl = ltl.implication %s, %np : !ltl.property
%res = ltl.not %impl : !ltl.property  
```

- **`s #=# p`**:  
```mlir
%np = ltl.not %p : !ltl.property	
%ds = ltl.delay %s, 1, 0 : !ltl.sequence
%c1 = hw.constant 1 : i1
%ant = ltl.concat %ds, c1 : !ltl.sequence 
%impl = ltl.implication %ant, %np : !ltl.property
%res = ltl.not %impl : !ltl.property  
```

- **`strong(s)`**: default for coverpoints, not supported in other cases.
- **`weak(s)`**: default for assert and assume, not supported for cover.  

- **`nexttime p`**:   
```mlir
ltl.delay %p, 1, 0 : !ltl.sequence   
```  

- **`nexttime[n] p`**:  
```mlir
ltl.delay %p, n, 0 : !ltl.sequence   
```  

- **`s_nexttime p`**: not really distinguishable from the weak version in CIRCT.
- **`s_nexttime[n] p`**: not really distinguishable from the weak version in CIRCT.

- **`always p`**:   
```mlir
ltl.repeat %p, 0 : !ltl.sequence   
```  

- **`always[n:m] p`**:   
```mlir
ltl.repeat %p, n, m : !ltl.sequence 
```  
- **`s_always[n:m] p`**: not really distinguishable in CIRCT  

- **`s_eventually p`**:   
```mlir
ltl.eventually %p : !ltl.property  
```

- **`eventually[n:m] p`**: not yet encodable in CIRCT.

- **`s_eventually[n:m] p`**: not yet encodable in CIRCT.

- **`p1 until p2`**:   
```mlir
%1 = ltl.until %p1, %p2 : !ltl.sequence  
```  

- **`p1 s_until p2`**: not really distinguishable from the weak version in CIRCT.

- **`p1 until_with p2`**: Equivalent to `(p1 until p2) implies (p1 and p2)`  
```mlir
%1 = ltl.until %p1, %p2 : !ltl.sequence
%2 = ltl.and %p1, %p2 : !ltl:property
%n1 = ltl.not %1 : !ltl.property
%res = ltl.or %n1, %2 : !ltl.property  
```  

- **`p1 s_until_with p2`**: not really distinguishable from the weak version in CIRCT.

> We don't yet support abort constructs but might in the future.  
> - **`accept_on ( expr ) p`**   
> - **`reject_on ( expr ) p`**  
> - **`sync_accept_on ( expr ) p`**   
> - **`sync_reject_on ( expr ) p`**   
  

## Representing the LTL Formalism


### Next / Delay

The `ltl.delay` sequence operation represents various shorthands for the *next*/**X** operator in LTL:

| Operation            | LTL Formula                 |
|----------------------|-----------------------------|
| `ltl.delay %a, 0, 0` | a                           |
| `ltl.delay %a, 1, 0` | **X**a                      |
| `ltl.delay %a, 3, 0` | **XXX**a                    |
| `ltl.delay %a, 0, 2` | a ∨ **X**a ∨ **XX**a        |
| `ltl.delay %a, 1, 2` | **X**(a ∨ **X**a ∨ **XX**a) |
| `ltl.delay %a, 0`    | **F**a                      |
| `ltl.delay %a, 2`    | **XXF**a                    |


### Until and Eventually

`ltl.until` is *weak*, meaning the property will hold even if the trace does not contain enough clock cycles to evaluate the property. `ltl.eventually` is *strong*, where `ltl.eventually %p` means `p` must hold at some point in the trace.


### Concatenation and Repetition

The `ltl.concat` sequence operation does not have a direct equivalent in LTL. It builds a longer sequence by composing multiple shorter sequences one after another. LTL has no concept of concatenation, or a *"v happens after u"*, where the point in time at which v starts is dependent on how long the sequence u was.

For a sequence u with a fixed length of 2, concatenation can be represented as *"(u happens) and (v happens 2 cycles in the future)"*, u ∧ **XX**v. If u has a dynamic length though, for example a delay between 1 and 2, `ltl.delay %u, 1, 1` or **X**u ∨ **XX**u in LTL, there is no fixed number of cycles by which the sequence v can be delayed to make it start after u. Instead, all different-length variants of sequence u have to be enumerated and combined with a copy of sequence v delayed by the appropriate amount: (**X**u ∧ **XX**v) ∨ (**XX**u ∧ **XXX**v). This is basically saying "u delayed by 1 to 2 cycles followed by v" is the same as either *"u delayed by 1 cycle and v delayed by 2 cycles"*, or *"u delayed by 2 cycles and v delayed by 3 cycles"*.

The *"v happens after u"* relationship is crucial to express sequences efficiently, which is why the LTL dialect has the `ltl.concat` op. If sequences are thought of as regular expressions over time, for example, `a(b|cd)` or *"a followed by either (b) or (c followed by d)"*, the importance of having a concatenation operation as temporal connective becomes apparent. Why LTL formalisms tend to not include such an operator is unclear.

As for `ltl.repeat`, it also relies on the semantics of *v happens after u* to compose the repeated sequences. Unlike `ltl.concat`, which can be expanded by LTL operators within a finite formula size, unbounded repetition cannot be expanded by listing all cases. This means unbounded repetition imports semantics that LTL cannot describe. The LTL dialect has this operation because it is necessary and useful for regular expressions and SVA.


## Types


### Overview

The `ltl` dialect operations defines two main types returned by its operations: sequences and properties. These types form a hierarchy together with the boolean type `i1`:

- a boolean `i1` is also a valid sequence
- a sequence `!ltl.sequence` is also a valid property

```
i1 <: ltl.sequence <: ltl.property
```

The two type constraints `AnySequenceType` and `AnyPropertyType` are provided to implement this hierarchy. Operations use these constraints for their operands, such that they can properly accept `i1` as a sequence, `i1` or a sequence as a property. The return type is an explicit `!ltl.sequence` or `!ltl.property`.

[include "Dialects/LTLTypes.md"]


## Operations

[include "Dialects/LTLOps.md"]
