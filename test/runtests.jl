using SyncSort
using Test

@testset "SyncSort.jl" begin
    # Write your tests here.
    x_float = rand(100)
    x_int = rand(Int, 100)

    syncsort!(x_float)
    syncsort!(x_int)

    @test issorted(x_int)
    @test issorted(x_float)
    
end
