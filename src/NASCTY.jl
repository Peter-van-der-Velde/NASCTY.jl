module NASCTY

using Dates
using ProgressMeter
using JLD2

# Write your package code here.
include("helper.jl")

include("data.jl")
export splittraces
export reshapetraces
export minmaxscale
export undersamplewithremainder
export loadASCADsampledata

include("genome.jl")
export ConvBlockGenome
export Core
export DenseLayerGenome
export Genome
export HyperParameters
export Main
export NASCTYGENOMEHYPERPARAMETERS
export NASCTYJLGENOMEHYPERPARAMETERS
export PoolingLayerGenome
export designspacesize
export nullpoolinglayer
export outputsize
export randomconvblockgenome
export randomdenselayergenome
export randomgenome
export randompoolinglayergenome
export phenotype

include("training.jl")
export fitnessevaluation

include("geneticalgorithm.jl")
export initpopulation
export evaluatepopulation
export tournamentselection
export produceoffspring

function nascty(hyperparameters::HyperParameters, training_data, validation_data, output_path="./$(now())_NASCTY.jld2")
    @info "Initialising population..."
    INPUTSIZE, _ = size(training_data |> first)
    population = initpopulation(NASCTY.randomgenome(hyperparameters, INPUTSIZE), hyperparameters.population_size)

    @info "Evaluate initial population gen 0."
    population = evaluatepopulation(population, fitnessevaluation)

    populations = [population] # the list of populations for each generation

    # preparing population splits
    parent_size = hyperparameters.population_size ÷ 2
    offspring_size = hyperparameters.population_size - parent_size

    @showprogress for gen in 1:hyperparameters.max_generation
        # @info "evaluating $gen..."
        sleep(0.25)

        # @info "Selecting parents..."
        parents = tournamentselection(population, parent_size, hyperparameters.tournament_size)

        # @info "Creating offspring..."
        offspring = produceoffspring(hyperparameters, parents, offspring_size, INPUTSIZE)

        # @info "Replace existing population..."
        population = [parents; offspring]

        # Update population records
        push!(populations, population)

        jldopen("$(output_path).jld2", "w+") do file
		    file["populations"] = populations
        end
    end
end



end
