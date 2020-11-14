#Analysis for function reuse.
* The function must have all arguments of primitive types.
* We need the worst latency of the function.
* The two instances of the function must not overlap.
* If the two instances occur before the yield then they can be shared.
* If the two instances occur on two sides of yield
