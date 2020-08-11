# Getting Started

## Installation

```
pkg> add https://github.com/CNCLgithub/GenRFS
```

## Example

This work generalizes common examples of random finite sets (RFS) such as poission multi-bernoulli RFS and presents a common API to build a spec of an arbritrary collection of [`RandomFiniteElement`](@ref).
Below we define such a random finite set using the type alias [`RFSElements`](@ref).

```@example 1
using Random
Random.seed!(1234)
using Gen
using GenRFS
# Here we are defining a simple poisson multi-bernoulli
pmbrfs = RFSElements{Float64}(undef, 2) # will have 2 rfe's
r1 = 3
p1 = PoissonElement{Float64}(r1, normal, (0., 1.0))
r2 = 0.4
b1 = BernoulliElement{Float64}(r2, uniform, (-1.0, 1.0))
pmbrfs[1] = p1
pmbrfs[2] = b1
pmbrfs
```

We can then treat `pmbrfs` as an argument to the [`rfs`](@ref) random variable using [`Gen.Distribution`](@ref) syntax.

```@example 1

xs = rfs(pmbrfs)
```

The loglikelihood of this sample is computing with partial memoization.

!!! note
    Due to the memoization of the partition table, the first call to `Gen.logpdf` may be slow

```@example 1
Gen.logpdf(rfs, xs, pmbrfs)
```

In addition, we can access to the different data correspondences present in calculation of the likelihood with [`AssociationRecord`](@ref).

```@example 1
record = AssociationRecord(5) # returning the top 5 correspondences
Gen.logpdf(rfs, xs, pmbrfs, record)
Dict(zip(record.table, record.logscores))
```


