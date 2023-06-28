using SyncSort
using Test
using Random

@testset "CPU single vectors" begin
    x = rand(100)
    syncsort!(x)
    @test issorted(x)

    x = rand(Int, 100)
    syncsort!(x)
    @test issorted(x)

    x = rand(UInt, 100)
    syncsort!(x)
    @test issorted(x)

    x = rand(Int, 100) |> Vector{Integer}
    syncsort!(x)
    @test issorted(x)
end


@testset "CPU double vectors" begin
    Random.seed!(0)
    x1 = rand(100)
    y1 = rand(100)
    x2 = copy(x1)
    y2 = copy(y1)
    syncsort!(x1, y1)
    @test issorted(x1)
    @test y1 == y2[sortperm(x2)]

    x1 = rand(Int, 100)
    x2 = copy(x1)
    y1 = rand(100)
    y2 = copy(y1)
    syncsort!(x1, y1)
    @test issorted(x1)
    @test y1 == y2[sortperm(x2)]

    x1 = rand(UInt, 100)
    x2 = copy(x1)
    y1 = rand(100)
    y2 = copy(y1)
    syncsort!(x1, y1)
    @test issorted(x1)
    @test y1 == y2[sortperm(x2)]

    x1 = rand(Int, 100) |> Vector{Integer}
    x2 = copy(x1)
    y1 = rand(100)
    y2 = copy(y1)
    syncsort!(x1, y1)
    @test issorted(x1)
    @test y1 == y2[sortperm(x2)]
end


@testset "CPU triple vectors" begin
    Random.seed!(0)
    x1 = rand(100)
    y1 = rand(100)
    z1 = rand(100)
    x2 = copy(x1)
    y2 = copy(y1)
    z2 = copy(z1)
    syncsort!(x1, y1, z1)
    @test issorted(x1)
    @test y1 == y2[sortperm(x2)]
    @test z1 == z2[sortperm(x2)]

    x1 = rand(Int, 100)
    y1 = rand(100)
    z1 = rand(100)
    x2 = copy(x1)
    y2 = copy(y1)
    z2 = copy(z1)
    syncsort!(x1, y1, z1)
    @test issorted(x1)
    @test y1 == y2[sortperm(x2)]
    @test z1 == z2[sortperm(x2)]

    x1 = rand(UInt, 100)
    y1 = rand(100)
    z1 = rand(100)
    x2 = copy(x1)
    y2 = copy(y1)
    z2 = copy(z1)
    syncsort!(x1, y1, z1)
    @test issorted(x1)
    @test y1 == y2[sortperm(x2)]
    @test z1 == z2[sortperm(x2)]

    x1 = rand(Int, 100) |> Vector{Integer}
    y1 = rand(100)
    z1 = rand(100)
    x2 = copy(x1)
    y2 = copy(y1)
    z2 = copy(z1)
    syncsort!(x1, y1, z1)
    @test issorted(x1)
    @test y1 == y2[sortperm(x2)]
    @test z1 == z2[sortperm(x2)]
end
