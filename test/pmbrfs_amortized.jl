using MOT
using GenRFS

xs = [10.0, 1.0, 0.0]

### PPP part
rate = 2
spatial = normal
spatial_params = (10.0, 1.0)
ppp_params = PPPParams(rate, spatial, spatial_params)


### MBRFS part
rs = [0.5, 0.9, 0.9]
rvs = fill(normal, 3)
rvs_args = [(1.0, 1.0),
            (0.0, 1.0),
            (2.0, 1.0)]

mbrfs_params = MBRFSParams(rs, rvs, rvs_args)

### PMBRFS: putting PPP and MBRFS together
saved_td = SavedTD([], [], [])
pmbrfs_params = PMBRFSParams(ppp_params, mbrfs_params, saved_td)

println(rs)
println(rvs)
println(rvs_args)
println(Gen.logpdf(pmbrfs, xs, pmbrfs_params))
println(pmbrfs_params.saved_td)
