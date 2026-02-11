import Flux

# include("hyperparameters.jl")
# include("genome.jl")

function fitnessevaluation(hyperparameters::HyperParameters, genome::Genome, training_data, validation_data, tracelength)
    if genome.loss != Inf32
        return genome.loss
    end

    model = genome |> g -> phenotype(g, tracelength) |> Flux.gpu
	opt = Flux.setup(Flux.Adam(5e-3), model)
	loss(m, x, y) = Flux.crossentropy(m(x), y)

	for epoch in 1:hyperparameters.n_epochs
		Flux.train!(loss, model, training_data, opt)
	end

	fitness = loss(model, validation_data[1], validation_data[2])

	return fitness
end

fakefitnessevaluation(_genome::Genome) = 5.0 + rand(1:100) / 100
