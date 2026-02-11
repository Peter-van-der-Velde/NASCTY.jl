import Flux

include("hyperparameters.jl")

struct PoolingLayerGenome
	type::Symbol
	size::Integer          # f, 2 to 50 in a step of 1
	stride::Integer        # s, 2 to 50 in a step of 1

	function PoolingLayerGenome(hyperparameters::HyperParameters, type, size, stride)
		# @info "plg" size stride
		@assert size ∈ hyperparameters.pooling_size
		@assert stride ∈ hyperparameters.pooling_stride

		new(type, size, stride)
	end
end

"""
    outputsize(inputsize::Integer, windowsize::Integer, stride::Integer)::Integer

Calculates the outputsize of a PoolingLayerGenome based upon its parameters.

```math
o = ⌊(|x| - |w|) ÷ s ⌋ + 1
```
where:\
`o`   = outputsize
`|x|` = inputsize
`|w|` = windowsize of a poolinglayer
`s`   = stride of a poolinglayer
"""
outputsize(inputsize::Integer, windowsize::Integer, stride::Integer)::Integer =
	floor((inputsize - windowsize) / stride) + 1

"""
    randompooling(hyperparameters::HyperParameters, inputsize::Integer, minoutputsize::Integer)::PoolingLayerGenome

Is a function that generates a random pooling layer genome, that satisfies the following
requirement. Its output must be higher or equal to `minoutputsize`.
Therefore to be considered an acceptable layer its output must fulfil the following formula.

```math
g(l) ≧ ⌊(|x| - f) ÷ s ⌋ + 1
```

where:\
`g(l)`= the minimal layer output size\
`|x|` = length of the input\
`f`   = pooling layer window size, a random value between 2 and `W_max`\
`s`   = pooling layer stride, a random value between 2 and `S_max`\

We can find `s` by
```math
S_max = ⌊(|x| - 2) ÷ (g(l) - 1)⌋
```
where:\
`W_min` = the lowest value the pooling window can have 2

We find the maximum value
that we could use for `w`.

```math
maxw = (|x| ) - o × s_min
```

"""
function randompoolinglayergenome(hyperparameters::HyperParameters,
    inputsize, minoutputsize)::PoolingLayerGenome

    # s = ⌊(l - f) / (g(l) - 1)⌋
    (minpoolsize, maxpoolsize) = hyperparameters.pooling_size |> extrema
    maxfiltersize = hyperparameters.convolution_kernel_size |> maximum

	maxpoolstride = minoutputsize != 1 ?
                    floor((inputsize - minpoolsize) / (minoutputsize - 1)) |> Integer :
                    maxfiltersize

    stride_candidates = filter(x -> x ≤ maxpoolstride, hyperparameters.pooling_stride)
	poolstride = stride_candidates |> rand
    @assert (stride_candidates |> length) >= 1

	# f = l - (gl - 1) * s
	max_window_size = inputsize - (minoutputsize - 1) * poolstride
	max_window_size = min(max_window_size, maxpoolsize)

    @assert minpoolsize <= max_window_size "$(minpoolsize) <= $(max_window_size)"
    windowsize_candidates = filter(x -> x ≤ max_window_size, hyperparameters.pooling_size)
    window_size = windowsize_candidates |> rand

    type = hyperparameters.pooling_type |> rand

    return PoolingLayerGenome(hyperparameters, type, window_size, poolstride)
end

nullpoolinglayer(hyperparameters)::PoolingLayerGenome =
	PoolingLayerGenome(hyperparameters, :null, 0, 0)

"""
Convolution block as part of the CNN.

The numbers have been taken from the NASCTY paper.
"""
struct ConvBlockGenome
	nfilters::Integer          # k, 2 to 128 in a step of 1
	filtersize::Integer        # f, 1 to 50 in a step of 1
	batchnormlayer::Bool
	poolinglayer::PoolingLayerGenome

	function ConvBlockGenome(hyperparameters::HyperParameters, nfilters, filtersize,
        batchnormlayer, poolinglayer)

		@assert nfilters ∈ hyperparameters.n_convolution_filters "nfilters ($nfilters) is not in $(hyperparameters.n_convolution_filters)"
        @assert filtersize ∈ hyperparameters.convolution_kernel_size "filtersize ($filtersize) is not in $(hyperparameters.convolution_kernel_size)"

        new(nfilters, filtersize, batchnormlayer, poolinglayer)
	end
end

function randomconvblockgenome(hyperparameters::HyperParameters, inputsize, minoutputsize = 2)
    nfilters = hyperparameters.n_convolution_filters |> rand
    filtersize = hyperparameters.convolution_kernel_size |> rand
    batchnormlayer = rand(1:2) % 2 == 0
    poolinglayer = randompoolinglayergenome(hyperparameters, inputsize, minoutputsize)

    pooling_layer_output = outputsize(inputsize, poolinglayer.size, poolinglayer.stride)

    if pooling_layer_output < minoutputsize
		@error "randomconvblock: unexpected output size is larger than minimum required" (pooling_layer_output < minoutputsize)

		return randomconvblockgenome(inputsize, minoutputsize)
	end

    return ConvBlockGenome(hyperparameters, nfilters, filtersize, batchnormlayer, poolinglayer)
end

struct DenseLayerGenome
	ndenseneurons::Integer # 1 to 20 in a step of 1

	function DenseLayerGenome(hyperparameters::HyperParameters, ndenseneurons)
		@assert ndenseneurons ∈ hyperparameters.n_neurons

		new(ndenseneurons)
	end
end

function randomdenselayergenome(hyperparameters::HyperParameters)
        ndenseneurons = hyperparameters.n_neurons |> rand
        DenseLayerGenome(hyperparameters, ndenseneurons)
end

struct Genome
	convblocks::Vector{ConvBlockGenome}
	denselayers::Vector{DenseLayerGenome}
	poolinglayer::PoolingLayerGenome

    loss::Float32
    # optimisation_function::Symbol
    # learning_rate::Float32

	function Genome(hyperparameters::HyperParameters
                    , convblocks::Vector{ConvBlockGenome}
					, denselayers::Vector{DenseLayerGenome}
					, poolinglayer::PoolingLayerGenome
                    , loss=Inf32)

		@assert (convblocks |> length) ∈ hyperparameters.n_convolution_layers "$(convblocks |> length) ∈ $(hyperparameters.n_convolution_layers)"
		@assert (denselayers |> length) ∈ hyperparameters.n_dense_layers "$(convblocks |> length) ∈ $(hyperparameters.n_dense_layers)"

		new(convblocks, denselayers, poolinglayer, loss)
	end

    function Genome(genome::Genome, loss)

		new(genome.convblocks, genome.denselayers, genome.poolinglayer, loss)
	end
end

"""
    randomgenome(inputsize::Integer = 700)::Genome

Generate a randomly initialised genome for a CNN.
The input size ensures that the pooling layer window does not exceed the input size.
"""
function randomgenome(hyperparameters::HyperParameters, inputsize)::Genome

	nconvblocks = hyperparameters.n_convolution_layers |> rand
	ndenselayers = hyperparameters.n_dense_layers |> rand
	convblocks = Vector{ConvBlockGenome}(undef, nconvblocks)

	minoutputsize = nconvblocks != 0 ?
                    2^(nconvblocks - 1) |> Integer :
                    1

	# @info "rg minoutputsize" minoutputsize nconvblocks

	newinputsize = inputsize

	for i in 1:nconvblocks
		# @info "rg" i newinputsize minoutputsize
		rcb = randomconvblockgenome(hyperparameters, newinputsize, minoutputsize)


		# oldinputsize = newinputsize
		newinputsize = outputsize(newinputsize, rcb.poolinglayer.size, rcb.poolinglayer.stride)
		# With a minimal stride of 2 the minimal needed
		# outputsize halves with each block.
		minoutputsize = minoutputsize ÷ 2

        convblocks[i] = rcb
	end

	denselayers = map(_ -> randomdenselayergenome(hyperparameters), 1:ndenselayers)
	# Always initialise the random pooling layer even if not used in the phenotype
	poolinglayer = randompoolinglayergenome(hyperparameters ,inputsize, 2)

	return Genome(hyperparameters, convblocks, denselayers, poolinglayer)
end

"""
    fixpoolinglayers(hyperparameters::HyperParameters, G::Genome, inputsize::Integer)

Adjusts pooling layer parameters (size and stride) across the genome to ensure the
signal dimension never drops below 1 before reaching the dense layers.
"""
function fixpoolinglayers(hyperparameters::HyperParameters, G::Genome, inputsize::Integer)::Genome
    # Standard CNN output size formula
    calc_out(l_in, f, s) = floor(Int, (l_in - f) / s) + 1
    # Required input to achieve a certain output
    calc_in_req(f, s, l_out) = (l_out - 1) * s + f

    function validmax(list, limit)
        allowed = filter(v -> v <= limit, list)

        return maximum(allowed)
    end

    function fixlayer(pl::PoolingLayerGenome, target_out_size, available_in_size)
        # Handle cases where available_in_size is already too small
        min_f = minimum(hyperparameters.pooling_size)
        safe_in = max(available_in_size, min_f + 1)

        if target_out_size > 1
            absolute_min_f = minimum(hyperparameters.pooling_size)

            max_s_possible = (safe_in - absolute_min_f) / (target_out_size - 1)

            new_stride = validmax(hyperparameters.pooling_stride, floor(Int, max_s_possible))

            max_f_possible = safe_in - (target_out_size - 1) * new_stride
            new_size = validmax(hyperparameters.pooling_size, floor(Int, max_f_possible))
        else
            new_stride = pl.stride
            new_size = validmax(hyperparameters.pooling_size, safe_in)
        end

        return PoolingLayerGenome(hyperparameters, pl.type, Int(new_size), Int(new_stride))
    end

    ps = [cb.poolinglayer for cb in G.convblocks]
    if isempty(ps) return G end

    layer_budgets = fill(0, length(ps))
    temp_l = inputsize
    for i in 1:length(ps)
        layer_budgets[i] = temp_l

        temp_l = calc_out(temp_l, minimum(hyperparameters.pooling_size), minimum(hyperparameters.pooling_stride))
        temp_l = max(temp_l, 1)
    end

    fixed_ps = Vector{PoolingLayerGenome}(undef, length(ps))
    current_target_out = 1

    for i in reverse(1:length(ps))
        pl_fixed = fixlayer(ps[i], current_target_out, layer_budgets[i])
        fixed_ps[i] = pl_fixed

        current_target_out = calc_in_req(pl_fixed.size, pl_fixed.stride, current_target_out)
    end

    new_convblocks = [
        ConvBlockGenome(
            hyperparameters,
            cb.nfilters,
            cb.filtersize,
            cb.batchnormlayer,
            fixed_ps[i]
        ) for (i, cb) in enumerate(G.convblocks)
    ]

    new_genome = Genome(hyperparameters, new_convblocks, G.denselayers, G.poolinglayer)

    # Final Validation
    function validateoutputsize(gen, insize)
        curr = insize
        for cb in gen.convblocks
            p = cb.poolinglayer
            curr = floor(Int, (curr - p.size) / p.stride) + 1
        end
        return curr
    end

    final_out = validateoutputsize(new_genome, inputsize)
    if final_out < 1
        @warn " Genome fix failed: Output size is $final_out. Model will not be viable."
    end

    return new_genome
end

"""
    phenotype(genome::Genome, inputsize, outputsize=256, seed=6363)

Constructs a `Flux.Chain` neural network (the phenotype) based on a defined `Genome`.

# Arguments
- `genome::Genome`: A struct containing the architectural hyperparameters (convblocks, denselayers, etc.).
- `inputsize`: The length of the 1D trace.
- `outputsize`: The number of output classes.
- `seed`: Reproducible weight initialisation.

# Returns
- `Flux.Chain`: A ready-to-train Flux model.
"""
function phenotype(genome::Genome, inputsize, outputsize = 256, seed = 6363)
	# Seed the random input so that all models are initialized with similar variables
	rng = Flux.Xoshiro(seed)
	chain = []
	nxs = inputsize
	nys = 1

	for cb in genome.convblocks
		conv = Flux.Conv((cb.filtersize,), nys => cb.nfilters, pad=Flux.SamePad(), init=Flux.kaiming_uniform(rng))
		nys = cb.nfilters |> Int64

		push!(chain, conv)

		if cb.batchnormlayer
			push!(chain, Flux.BatchNorm(nys))
		end

		pl = cb.poolinglayer
		if pl.type == :average
			push!(chain, Flux.MeanPool((pl.size |> Int64,); stride = (pl.stride |> Int64,)))
		else
			push!(chain, Flux.MaxPool((pl.size |> Int64,); stride = (pl.stride |> Int64,)))
		end
		nxs = ceil((nxs + 1 - pl.size) / pl.stride) |> Integer
	end

	# add a pooling layer if there are no convlayers
	if length(genome.convblocks) == 0
		pl = genome.poolinglayer
		if pl.type == :average
			push!(chain, Flux.MeanPool((pl.size |> Int64,); stride = (pl.stride |> Int64,)))
		else
			push!(chain, Flux.MaxPool((pl.size |> Int64,); stride = (pl.stride |> Int64,)))
		end
		# update size of
		nxs = ceil((nxs + 1 - pl.size) / pl.stride) |> Integer
	end

	# Add flatten layer.
	push!(chain, MLUtils.flatten)

	nxs = nxs * nys
	nys = 1

	for dl in genome.denselayers
		dense = Flux.Dense(nxs => dl.ndenseneurons, Flux.selu, init=Flux.kaiming_uniform(rng))
		push!(chain, dense)
		nxs = dl.ndenseneurons
	end
	push!(chain, Flux.Dense(nxs => outputsize; init=Flux.zeros32))

	push!(chain, Flux.softmax)

	Flux.Chain(chain)
end

"""
    nparameters(G::Genome)

Calculate the structural complexity of a `Genome` by counting its active hyperparameters.

# Arguments
- `G::Genome`

# Returns
- `Int`: The total count of structural hyperparameters.
"""
function nparameters(G::Genome)
	n_parameters_pooling = 3
	n_parameters_conv = 3 + n_parameters_pooling
	n_parameters_dense = 1

	n_c = length(G.convblocks) * n_parameters_conv
	n_d = length(G.denselayers) * n_parameters_dense

	if n_c == 0
		n_parameters_pooling + n_d
	else
		n_c + n_d
	end
end

