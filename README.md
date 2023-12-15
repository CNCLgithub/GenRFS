# GenRFS

> This is an unofficial extension of Gen to handle Random finite sets. Work in progress


## Installation

``` julia
Pkg > add https://github.com/CNCLgithub/GenRFS.git
```


## Usage


### Building blocks

Random finite sets (RFS) are distributions over sets.
In `GenRFS`, these distributions are parameterized with a set (technically a `Vector`) of Random finite elements (RFEs).

Each element takes the form of `RandomFiniteElement{T}(c, d, args)` and can sample sets (implemented as `Vector`) of type `T`.
The cardinality of these samples is defined as a distribution parameterized by `c` and the content is drawn from an internal distribution `d` with parameters `args`.

Take for example a `BernoulliElement`, `b`, that can either sample and empty set `{}` or a set of one thing `{x}` with probability `0.5`. 
Here `x` is drawn from an internal distribution, parameterized as a normal.

> While not currently supported, it is possible to define RFEs where content and cardinality are dependent. 

``` julia
using Gen
using GenRFS

b = BernoulliElement{Float64}(0.5, normal, (0., 1.))
```


There are other common kinds of RFEs such as the `PoissonElement`.
A single `PoissonElement` is equivalent to a Poisson Point process. 
``` julia

p = PoissonElement{Float64}(3, uniform, (-1., 1.0))
```


However, these are just elements, and alone do not define an RFS.
Distinct from previous approaches, an arbitrary collection of RFEs can be accumulated to define an RFS.

First we can define a type safe instance of `RFS{T} <: Gen.Distribution{T}`
> This is useful for runtime but not neccessary as a generic instance is provided with `rfs` whichi is of `RFS{Any} <: Gen.Distribution{Any}`.

``` julia
const rfs_float = RFS{Float64}()
```


Then we can define what would normally be called a Bernoulli Random Finite set (BRFs) and use Gen's methods for sampling and logpdf.
``` julia

brfs = [b]
xs = rfs_float(brfs)
ls = Gen.logpdf(rfs_float, xs, brfs)
```


Here are more examples of common RFS
``` julia
# multi bernoulli rfs
b2 = BernoulliElement{Float64}(0.8, normal, (1., 0.1))
mbrfs = [b, b2]
xs = rfs(brfs)
ls = Gen.logpdf(rfs, xs, mbrfs)


# poisson multi bernoulli rfs
pmbrfs = [b, b2]
xs = rfs(brfs)
ls = Gen.logpdf(rfs, xs, pmbrfs)
```

You can also define more exotic RFS
``` julia

poissons = Vector{RandomFiniteElement{Float64}}(undef, 4)
for i = 1:4
    poissons[i] = PoissonElement{Float64}(i, normal, (i*2.0, 1.0))
end
xs = rfs(poissons)
ls = Gen.logpdf(rfs, xs, poissons)
```

Unfortunately, the analytical algorithm for the logpdf has factorical complexity.
The good news is that in most scenarios, only a handful of partitions contain most of the mass.

To take advantage of this, you can use the `MRFS{T}` random variable that uses a random walk procedure to sample a subset of high mass partitions with quadratic 


``` julia
mrfs_float = MRFS{Float64}()
steps = 20 # number of steps for random walk
temp = 1.0 # temperature of kernel; 1.0 -> +Inf is choatic, 1.0 -> 0. is greedy
xs = mrfs_float(poissons, steps, temp)
ls = Gen.logpdf(mrfs_float, xs, poissons, steps, temp)
```

You can empirically evaluate the coverage of the entire partition mass space to verify steps and temp. 

``` julia
TODO
```


### Basic usage

You can use `RFS{T}` and `MRFS{T}` anywhere you can  `Gen.Distribitions{T}`.

``` julia
@gen static function foo()
    r = @trace(beta(2., 2.), :r)
    mu = @trace(uniform(-3., 3.), :mu)
    b = BernoulliElement{Float64}(r, normal, (mu, 1.0))
    xs = @trace(mrfs_float([b]), :xs)
    return xs
end
```

### Incremental computation
However, the `Gen.Distribution` API lacks incremental computation (by design).
To support incremental computation, use `GenRFS.RFGM` that implements `Gen.GenerativeFunction` over `GenRFS.AbstractRFS`.


```julia 
# define generative function (estimator<:AbstractRFS, args::Tuple)
const rfgm = RFGM(MRFS{Float64}(), (100, 1.0))
# const rfgm = RFGM(RFS{Float64}(), ())
@gen static function bar(n::Int64)
    mus = @trace(Gen.Map(normal)(1:n), :mus)
    es = map(m -> BernoulliElement{Float64}(0.9, (m, 1.0)), mus)
    xs = @trace(rfgm(es), :xs)
end
```

Note that this uses a new trace type `RFSTrace` and has different choicemap structure (each sample is indexed like a `Gen.Map` combinator). Like the distribution api, the order of the indices does not affect the logscore but a supplied order is maintained for efficiency purposes. 

## TODOS

- package convergence analysis tools
- propagate gradients when possible
