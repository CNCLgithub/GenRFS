export convergence

using UnicodePlots
using BenchmarkTools

function convergence(xs::Vector{T}, es::RFSElements{T},
                     steps::Int, t::Float64) where {T}
    println("Time to execute rfs")
    @time full_ls, full_pt = associations(es, xs)

    println("Time to execute markov rfs")
    display(@benchmark associations($es, $xs, $steps, $t))
    m_ls, m_pt = associations(es, xs, steps, t)

    @show logsumexp(m_ls)
    @show logsumexp(full_ls)
    mass_conv = exp(logsumexp(m_ls) - logsumexp(full_ls))
    psize_conv = Float64(size(m_pt, 3)) / Float64(size(full_pt, 3))


    result = Dict(:mass_convergence => mass_conv,
                  :psize_conv => psize_conv)

    display(histogram(m_ls,
                    vertical = true,
                    width = 30,
                    title = "Approximation",
                    xlim = (minimum(full_ls),
                            maximum(full_ls))))
    display(histogram(full_ls, vertical = true,
                    bins = 50,
                    width = 30,
                    title = "P(xs | es)"))

    result
end
