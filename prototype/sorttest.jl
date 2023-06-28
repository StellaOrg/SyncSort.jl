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


function syncsortperm!(v, rest...)
    isorted = sortperm(v)

    # Permute elements of v and rest in-place following isorted
    done = firstindex(v) - 1
    i = firstindex(isorted)
    @inbounds while i <= lastindex(isorted)
        j = i
        js = isorted[j]
        if js != done
            @inbounds while true
                isorted[j] = done
                isorted[js] == done && break
                v[j], v[js] = v[js], v[j]
                for r in rest
                    r[j], r[js] = r[js], r[j]
                end
                j = js
                js = isorted[j]
            end
        end
        i += 1
    end
end


@inbounds function test3(n)
        
    x = rand(n)
    y = rand(n)
    z = rand(n)

    syncsortperm!(x,y,z)
    x,y,z
end


@inbounds function test4(n)
        
    x = rand(n)
    y = rand(n)
    z = rand(n)

    syncsort!(x,y,z; alg=SyncSort.CPUSort.ScratchQuickSort())
    x,y,z
end


println("\nsyncsort:")
display(@benchmark(test1(10_000))) 

println("\nsortperm:")
display(@benchmark(test2(10_000)))

println("\nsyncsortperm:")
display(@benchmark(test3(10_000)))

println("\nsyncsort/ScratchQuickSort:")
display(@benchmark(test4(10_000))) 
