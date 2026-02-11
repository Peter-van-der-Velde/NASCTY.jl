"""
    Hyperparameters

Hyperparameter search space configuration for neural architecture search (NAS) and
hyperparameter optimisation using genetic algorithms.

This struct defines the search space for optimising neural network architectures,
training parameters, and genetic algorithm settings. Each field contains a vector
of possible values that the genetic algorithm can explore.

# Fields

## MLP (Multi-Layer Perceptron) Parameters
- `n_dense_layers::Vector{Integer}`: Number of dense/fully-connected layers in the MLP.
  Example: `[1, 2, 3]` for searching between 1-3 dense layers.
- `n_neurons::Vector{Integer}`: Number of neurons in each dense layer.
  Example: `[32, 64, 128]` for 32, 64, or 128 neurons per layer.

## CNN (Convolutional Neural Network) Parameters
- `n_convolution_layers::Vector{Integer}`: Number of convolutional layers.
  Example: `[1, 2, 3]` for 1-3 convolutional layers.
- `n_convolution_filters::Vector{Integer}`: Number of filters/kernels in each
  convolutional layer. Example: `[16, 32, 64]`.
- `convolution_kernel_size::Vector{Integer}`: Size of convolutional kernels.
  Example: `[3, 5, 7]` for 3x3, 5x5, or 7x7 kernels.
- `pooling_size::Vector{Integer}`: Size of pooling windows.
  Example: `[2, 3]` for 2x2 or 3x3 pooling.
- `pooling_stride::Vector{Integer}`: Stride for pooling operations.
  Example: `[2, 3]` for stride of 2 or 3.
- `pooling_type::Vector`: Type of pooling operation (e.g., `[:max, :avg]`).
- `padding::Vector`: Padding method for convolutional layers (e.g., `[:same, :valid]`).
- `normalisation_batch_layer::Vector{Bool}`: Whether to use batch normalisation after layers.

## Network Training & Initialisation Parameters
- `optimisation_function`: Vector of optimisation algorithms (e.g., `[ADAM(), SGD()]`).
- `learning_rate::Vector`: Learning rate for training.
- `activation_function::Vector`: Neural network activation functions for hidden layers.
  Example: `[RELU(), TANH(), SIGMOID()]`.
- `layer_initialisation::Vector`: Weight initialisation methods for layers.
  Example: `[XAVIER(), KAIMING()]`.
- `softmax_initialisation::Vector`: Output layer initialisation/activation for
  classification tasks.

## Genetic Algorithm Hyperparameters
- `n_epochs::Integer`: Number of training epochs for evaluating each possible design.
- `batch_size::Vector`: Batch sizes for training.
- `polynomial_mutation::Integer`: Parameter for polynomial mutation operator in
  genetic algorithm. Controls mutation strength.
- `anti_stagnation_strategies::Vector`: Anti-stagnation strategies for combatting
  premature convergence.
"""
struct HyperParameters
    # MLP
    n_dense_layers::Vector{Integer}
    n_neurons::Vector{Integer}

    # CNN
    n_convolution_layers::Vector{Integer}
    n_convolution_filters::Vector{Integer}
    convolution_kernel_size::Vector{Integer}
    pooling_size::Vector{Integer}
    pooling_stride::Vector{Integer}
    pooling_type::Vector
    padding::Vector
    normalisation_batch_layer::Vector{Bool} # TODO: needed?

    # Phenotype training & initialisation
    optimisation_function::Vector # TODO: implement into genome
    learning_rate::Vector{Float32} # TODO: implement into genome
    activation_function::Vector # TODO: implement into genome
    layer_initialisation::Vector # TODO: implement into genome
    softmax_initialisation::Vector # TODO: implement into genome

    # Genetic algorithm hyperparameters
    population_size::Integer
    n_epochs::Integer
    max_generation::Integer
    batch_size::Vector # TODO: implement into genome
    tournament_size::Integer
    polynomial_mutation::Integer
    anti_stagnation_strategies::Vector
end

"""
    NASCTYGENOMEHYPERPARAMETERS()

This function returns a pre-configured `Hyperparameters` struct with defaults
taken from:

F. Schijlen, L. Wu, and L. Mariot, “Nascty: Neuroevolution to attack side-channel
leakages yielding convolutional neural networks,” Mathematics, vol. 11, no. 12, p. 2616,
2023, issn: 2227-7390. doi: [10.3390/math11122616](https://doi.org/10.3390/math11122616)

"""
function NASCTYGENOMEHYPERPARAMETERS()
    # MLP
    n_dense_layers = 0:1:5 |> collect
    n_neurons = 1:1:20 |> collect

    # CNN
    n_convolution_layers = 0:1:5 |> collect
    n_convolution_filters = 2:1:128 |> collect
    convolution_kernel_size = 1:1:50 |> collect
    pooling_size = 2:1:50 |> collect
    pooling_stride = 2:1:50 |> collect
    pooling_type = [:average, :max]
    normalisation_batch_layer = [true, false]

    padding = [:same]

    # Phenotype training & initialisation
    optimisation_function = [:adam]
    learning_rate = [5e-3]
    activation_function = [:selu]

    # KAIMING is also known as He initialisation.
    # see @doc Flux.kaiming_uniform for more information
    layer_initialisation = [:kaiming_uniform]
    softmax_initialisation = [:zeros32]

    # Genetic algorithm hyperparameters
    population_size = 100
    n_epochs = 10
    max_generation = 50
    tournament_size = 3
    batch_size = [100]
    polynomial_mutation = 20
    anti_stagnation_strategies = []

    return HyperParameters(
        n_dense_layers,
        n_neurons,
        n_convolution_layers,
        n_convolution_filters,
        convolution_kernel_size,
        pooling_size,
        pooling_stride,
        pooling_type,
        padding,
        normalisation_batch_layer,
        optimisation_function,
        learning_rate,
        activation_function,
        layer_initialisation,
        softmax_initialisation,
        population_size,
        n_epochs,
        max_generation,
        batch_size,
        tournament_size,
        polynomial_mutation,
        anti_stagnation_strategies
    )
end

"""
    NASCTYGENOMEHYPERPARAMETERS()

This function returns a pre-configured `Hyperparameters` struct with defaults
taken from:

F. Schijlen, L. Wu, and L. Mariot, “Nascty: Neuroevolution to attack side-channel
leakages yielding convolutional neural networks,” Mathematics, vol. 11, no. 12, p. 2616,
2023, issn: 2227-7390. doi: [10.3390/math11122616](https://doi.org/10.3390/math11122616)

and adds the partial_restart antistagnation strategy from:

Velde, Peter. Weighted, Weighted and Art Found Wanting: A Complexity-minimisation Approach for Neuroevolution-based Side-channel Analysis.
MS thesis. University of Twente, 2025.

"""
function NASCTYJLGENOMEHYPERPARAMETERS()
    # MLP
    n_dense_layers = 0:1:5 |> collect
    n_neurons = 1:1:20 |> collect

    # CNN
    n_convolution_layers = 0:1:5 |> collect
    n_convolution_layers = 2:1:128 |> collect
    n_convolution_filters = 2:1:128 |> collect
    convolution_kernel_size = 1:1:50 |> collect
    normalisation_batch_layer = [true, false]

    pooling_size = 2:1:50 |> collect
    pooling_stride = 2:1:50 |> collect
    pooling_type = [:average, :max]

    padding = [:same]

    # Phenotype training & initialisation
    optimisation_function = [:adam]
    learning_rate = [5e-3]
    activation_function = [:selu]

    # KAIMING is also known as He initialisation.
    # see @doc Flux.kaiming_uniform for more information
    layer_initialisation = [:kaiming_uniform]
    softmax_initialisation = [:zeros32]

    # Genetic algorithm hyperparameters
    population_size = 100
    n_epochs = 10
    max_generation = 50
    batch_size = [100]
    tournament_size = 3
    polynomial_mutation = 20
    anti_stagnation_strategies = [:partial_restart]

    return HyperParameters(
        n_dense_layers,
        n_neurons,
        n_convolution_layers,
        n_convolution_filters,
        convolution_kernel_size,
        pooling_size,
        pooling_stride,
        pooling_type,
        padding,
        normalisation_batch_layer,
        optimisation_function,
        learning_rate,
        activation_function,
        layer_initialisation,
        softmax_initialisation,
        population_size,
        n_epochs,
        max_generation,
        batch_size,
        tournament_size,
        polynomial_mutation,
        anti_stagnation_strategies
    )
end


"""
    designspacesize(hyperparameters::GenomeHyperparameters)

# Arguments
- `genome::GenomeHyperparameters`: hyperparameter configuration containing search space definitions.

# Returns
- `BigInt`: Total number of unique hyperparameter combinations possible (design space cardinality).

"""
function designspacesize(hyperparameters)::BigInt

    total = BigInt(1)

    total *= hyperparameters.n_dense_layers |> length
    total *= hyperparameters.n_neurons |> length
    total *= hyperparameters.n_convolution_layers |> length
    total *= hyperparameters.n_convolution_filters |> length
    total *= hyperparameters.convolution_kernel_size |> length
    total *= hyperparameters.pooling_size |> length
    total *= hyperparameters.pooling_stride |> length
    total *= hyperparameters.pooling_type |> length
    total *= hyperparameters.padding |> length
    total *= hyperparameters.normalisation_batch_layer |> length
    total *= hyperparameters.optimisation_function |> length
    total *= hyperparameters.learning_rate |> length
    total *= hyperparameters.activation_function |> length
    total *= hyperparameters.layer_initialisation |> length
    total *= hyperparameters.softmax_initialisation |> length
    total *= hyperparameters.batch_size |> length

    return total
end
