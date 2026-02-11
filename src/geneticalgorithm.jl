"""
   initpopulation(n = 100)

Initialise a random population of size n, using the initialisation function f.
"""
initpopulation(f, n=100) = map(_ -> f, 1:n)

populationbest(P⃗::Vector) = argmin(g -> g.loss, P⃗)

evaluatepopulation(P⃗::Vector, f::Function)::Vector = map(g -> NASCTY.Genome(g, f(g)), P⃗)

"""
    tournamentselection(P⃗::Vector, n_winners::Integer, tournament_size::Integer=3, f=x->x.loss)

Select `n_winners` from a population `P⃗` using tournament selection.

For each winner, `tournament_size` individuals are sampled at random (with replacement),
and the one with the minimum value returned by the fitness function `f` is chosen.

# Arguments
- `P⃗`: The population vector.
- `n_winners`: Total number of contestants to select.
- `tournament_size`: Number of contestants in each tournament (default: 3).
- `f`: A function to extract the fitness/cost value to be minimised.
"""
tournamentselection(P⃗::Vector, n_winners::Integer, tournament_size::Integer=3, f=x -> getfield(x, :loss)) =
    1:n_winners .|> _ -> rand(P⃗, tournament_size) |> tournament -> argmin(g -> g |> f, tournament)

"""
    crossover(A::Genome, B::Genome)

Performs a one-point crossover between two `Genome` objects.

Creates two new `Genome` objects by exchanging parts of the `convblocks` and `denselayers` attributes between the input genomes.
There is a 50% chance of swapping the `poolinglayer` attribute as well between the offspring.

# Arguments
- `A`: The first parent `Genome`.
- `B`: The second parent `Genome`.

# Returns
A tuple containing the two new `Genome` objects.
"""
function crossover(hyperparameters::HyperParameters, A::Genome, B::Genome)::Tuple{Genome,Genome}
    # Selecting the cut points in the genomes
    min_conv_layers = hyperparameters.n_convolution_layers |> minimum
    n = rand(min_conv_layers:(A.convblocks|>length))
    m = rand(min_conv_layers:(B.convblocks|>length))

    # Following the logic of the original NASCTY algorithm,
    # Any extra gene over the limit of 5 over given over to the void.
    max_conv_layers = hyperparameters.n_convolution_layers |> maximum
    convblocks_A′ = [take(A.convblocks, n); skip(B.convblocks, m)] |> x⃗ -> take(x⃗, max_conv_layers)
    convblocks_B′ = [take(B.convblocks, m); skip(A.convblocks, n)] |> x⃗ -> take(x⃗, max_conv_layers)

    min_dense_layers = hyperparameters.n_dense_layers |> minimum
    n = rand(min_dense_layers:(A.denselayers|>length))
    m = rand(min_dense_layers:(B.denselayers|>length))

    max_dense_layers = hyperparameters.n_dense_layers |> maximum
    denselayers_A′ = [take(A.denselayers, n); skip(B.denselayers, m)] |> x⃗ -> take(x⃗, max_dense_layers)
    denselayers_B′ = [take(B.denselayers, m); skip(A.denselayers, n)] |> x⃗ -> take(x⃗, max_dense_layers)

    pooling_layer_A′ = A.poolinglayer
    pooling_layer_B′ = B.poolinglayer

    switch_pooling_layer = rand(0:1) % 2 == 0
    if switch_pooling_layer
        pooling_layer_A′, pooling_layer_B′ = pooling_layer_B′, pooling_layer_A′
    end

    A′ = Genome(
        hyperparameters, convblocks_A′, denselayers_A′, pooling_layer_A′
    )

    B′ = Genome(
        hyperparameters, convblocks_B′, denselayers_B′, pooling_layer_B′
    )

    (A′, B′)
end

function mutateaddlayer(hyperparameters::HyperParameters, G::Genome, inputsize)::Genome
    cls = G.convblocks
    dls = G.denselayers
    pl = G.poolinglayer

    max_conv_layers = hyperparameters.n_convolution_layers |> maximum
    max_dense_layers = hyperparameters.n_dense_layers |> maximum

    option = if cls |> length >= max_conv_layers
        :dense
    elseif dls |> length >= max_dense_layers
        :conv
    else
        if rand(1:2) % 2 == 0 # choose dense or convolutional randomly
            :dense
        else
            :conv
        end
    end

    if option == :dense
        return Genome(hyperparameters, cls, [dls; randomdenselayergenome(hyperparameters)], pl)
    else # :conv
        return Genome(hyperparameters, [cls; randomconvblockgenome(hyperparameters, inputsize)], dls, pl)
    end
end

function mutateremovelayer(hyperparameters::HyperParameters, G::Genome)::Genome
    cls = G.convblocks
    dls = G.denselayers
    pl = G.poolinglayer

    dropat(x⃗::AbstractVector, i::Integer) = [x⃗[1:(i-1)]; x⃗[(i+1):end]]

    min_conv_layers, max_conv_layers = hyperparameters.n_convolution_layers |> extrema
    min_dense_layers, max_dense_layers = hyperparameters.n_dense_layers |> extrema

    if (dls |> length != min_dense_layers) && (cls |> length != min_conv_layers)
        n = cls |> length
        m = dls |> length

        i = rand(1:n)
        j = rand(1:m)

        m_cls = dropat(cls, i)
        m_dls = dropat(dls, j)

        if rand(1:2) % 2 == 0 # choose dense or convolutional randomly
            return Genome(hyperparameters, cls, m_dls, pl)
        else
            return Genome(hyperparameters, m_cls, dls, pl)
        end
    elseif cls |> length == min_conv_layers && (dls |> length != min_dense_layers)
        n = dls |> length
        i = rand(1:n)
        m_dls = dropat(dls, i)

        return Genome(hyperparameters, cls, m_dls, pl)
    elseif dls |> length == min_dense_layers && (cls |> length != min_conv_layers)
        n = cls |> length
        i = rand(1:n)
        m_cls = dropat(cls, i)

        return Genome(hyperparameters, m_cls, dls, pl)
    else # No available layers to delete
        return G
    end
end

function mutatepolynomial(hyperparameters::HyperParameters, G::Genome)::Genome
    # each parameter has chance 1 / n of being mutated
    p = 1 / (G |> nparameters)
    η = hyperparameters.polynomial_mutation

    δ(u, η) =
        if u < 0.5
            (2 * u)^(1 / (η + 1)) - 1
        else
            1 - (2 * (1 - u))^(1 / (η + 1))
        end

    # Polynomial mutation
    mutatepolynomial(x, x_min, x_max, η=20, u=rand()) =
        x + δ(u, η) * (x_max - x_min) |> x′ -> clamp(x′, x_min, x_max)

    function mutate_index(current_val, list, η)
        idx = findfirst(==(current_val), list)
        @assert idx != nothing "$(current_val) not found as a valid value in option $(list)"

        u = rand()
        # Mutate the index in continuous space
        new_idx_float = idx + δ(u, η) * (length(list) - 1)

        # Round and clamp to valid index range
        final_idx = clamp(round(Int, new_idx_float), 1, length(list))

        return list[final_idx]
    end

    maybedo(f, p, x, args...) =
        rand() <= p ? f(x, args...) : x

    mppool(pl::PoolingLayerGenome, p, η)::PoolingLayerGenome =
        PoolingLayerGenome(
            hyperparameters,
            maybedo(_ -> rand(hyperparameters.pooling_type), p, pl.type),
            maybedo(v -> mutate_index(v, hyperparameters.pooling_size, η), p, pl.size),
            maybedo(v -> mutate_index(v, hyperparameters.pooling_stride, η), p, pl.stride)
        )

    mpconv(cl::ConvBlockGenome, p, η)::ConvBlockGenome =
        ConvBlockGenome(
            hyperparameters,
            maybedo(v -> mutate_index(v, hyperparameters.n_convolution_filters, η), p, cl.nfilters),
            maybedo(v -> mutate_index(v, hyperparameters.convolution_kernel_size, η), p, cl.filtersize),
            maybedo(_ -> rand([true, false]), p, cl.batchnormlayer),
            mppool(cl.poolinglayer, p, η)
        )

    mpdense(dl::DenseLayerGenome, p, η)::DenseLayerGenome =
        DenseLayerGenome(
            hyperparameters,
            maybedo(v -> mutate_index(v, hyperparameters.n_neurons, η), p, dl.ndenseneurons)
        )

    if G.convblocks |> length > 0
        Genome(
            hyperparameters,
            map(x -> mpconv(x, p, η), G.convblocks),
            map(x -> mpdense(x, p, η), G.denselayers),
            G.poolinglayer
        )
    else
        Genome(
            hyperparameters,
            G.convblocks,
            map(x -> mpdense(x, p, η), G.denselayers),
            mppool(G.poolinglayer, p, η)
        )
    end
end

function mutate(hyperparameters::HyperParameters, G::Genome, inputsize)::Genome
    nc, nd = (G.convblocks |> length, G.denselayers |> length)

    min_conv_layers, max_conv_layers = hyperparameters.n_convolution_layers |> extrema
    min_dense_layers, max_dense_layers = hyperparameters.n_dense_layers |> extrema

    mutoptions = if nc == max_conv_layers && nd == max_dense_layers
        [:REMOVELAYER, :POLYNOMIALMUTATION]
    elseif nc == min_conv_layers && nd == min_dense_layers
        [:ADDLAYER, :POLYNOMIALMUTATION]
    else
        [:ADDLAYER, :REMOVELAYER, :POLYNOMIALMUTATION]
    end

    mutation_strategy = mutoptions |> rand
    mutated_genome = if (mutation_strategy == :ADDLAYER)
        mutateaddlayer(hyperparameters, G, inputsize)
    elseif (mutation_strategy == :REMOVELAYER)
        mutateremovelayer(hyperparameters, G)
    elseif (mutation_strategy == :POLYNOMIALMUTATION)
        mutatepolynomial(hyperparameters, G)
    else
        @error "Unsupported and unknown mutation strategy" mutation_strategy
    end

    mutated_genome
end

function produceoffspring(hyperparameters::HyperParameters, parents::Vector{Genome}, offspring_size::Integer, inputsize::Integer)::Vector{Genome}

    n_of_couples = offspring_size / 2 |> ceil |> Integer
    parent_couples = 1:n_of_couples .|> _ -> rand(parents, 2)

    newoffspring = parent_couples .|> parents -> crossover(hyperparameters, parents[1], parents[2])
    newoffspring = (Iterators.flatten(newoffspring)|>collect)[1:offspring_size]

    mutatedoffspring = newoffspring .|> ofp -> mutate(hyperparameters, ofp, inputsize)

    return mutatedoffspring .|> offspring -> fixpoolinglayers(hyperparameters, offspring, inputsize)
end
