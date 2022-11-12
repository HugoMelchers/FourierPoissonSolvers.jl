function which_fft(bcs, is_offset)
    if bcs isa Periodic
        DHT
    elseif is_offset
        if bcs isa Tuple{Dirichlet, Dirichlet}
            RODFT10
        elseif bcs isa Tuple{Dirichlet, Neumann}
            RODFT11
        elseif bcs isa Tuple{Neumann, Dirichlet}
            REDFT11
        elseif bcs isa Tuple{Neumann, Neumann}
            REDFT10
        end
    else
        if bcs isa Tuple{Dirichlet, Dirichlet}
            RODFT00
        elseif bcs isa Tuple{Dirichlet, Neumann}
            RODFT01
        elseif bcs isa Tuple{Neumann, Dirichlet}
            REDFT01
        elseif bcs isa Tuple{Neumann, Neumann}
            REDFT00
        end
    end
end

function inverse_transform_of(t)
    if t == DHT || t == RODFT00 || t == RODFT11 || t == REDFT00 || t == REDFT11
        t
    elseif t == RODFT01
        RODFT10
    elseif t == RODFT10
        RODFT01
    elseif t == REDFT01
        REDFT10
    elseif t == REDFT10
        REDFT01
    end
end

function scaling_factor(t, n)
    if t == DHT
        Float64(n)
    elseif t == REDFT01 || t == REDFT10 || t == REDFT11 || t == RODFT01 || t == RODFT10 || t == RODFT11
        Float64(2n)
    elseif t == REDFT00
        Float64(2n + 2)
    elseif t == RODFT00
        Float64(2n - 2)
    end
end

ev(l) = -2 + 2cos(pi*l)

function evs(t, n)
    ks = if t == DHT
        2fftfreq(n)
    elseif t == RODFT00
        (1:n) ./ (n + 1)
    elseif t == RODFT10
        (1:n) ./ n
    elseif t == REDFT00
        (0:n-1) ./ (n-1)
    elseif t == REDFT10
        (0:n-1) ./ n
    elseif t == RODFT01 || t == RODFT11 || t == REDFT01 || t == REDFT11
        ((1:n) .- 0.5) ./ n
    end
    ev.(ks)
end


