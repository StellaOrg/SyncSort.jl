using SyncSort

using BenchmarkTools

@inbounds function test1(n)
        
    x = rand(n)
    y = rand(n)
    z = rand(n)

    syncsort!(x,y,z)
    x,y,z
end

@inbounds function test2(n)
        
    x = rand(n)
    y = rand(n)
    z = rand(n)

    isorted = sortperm(x)
    x = x[isorted]
    y = y[isorted]
    z = z[isorted]

    x,y,z
end

println("syncsort")
display(@benchmark(test1(100_000))) 

println("sortperm")
display(@benchmark(test2(100_000)))
