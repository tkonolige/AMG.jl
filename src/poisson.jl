module Poisson

export poisson

using PyCall

@pyimport pyamg.gallery as gallery
@pyimport pyamg.gallery.diffusion as diffusion

function poisson(s)
    stencil = diffusion.diffusion_stencil_2d()
    m = gallery.stencil_grid(stencil, s, format="csc")
    plus_one(x) = map(x -> x + 1, x)
    SparseMatrixCSC(m[:shape][1], m[:shape][2], plus_one(m[:indptr]), plus_one(m[:indices]), m[:data])
end

end
