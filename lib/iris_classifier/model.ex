defmodule IrisClassifier.Model do
  import Nx.Defn

  def build_model do
    # Define the model using the input/dense syntax
    input = Axon.input("input", shape: {nil, 4})

    model =
      input
      |> Axon.dense(64)
      |> Axon.relu()
      |> Axon.dropout(rate: 0.2)
      |> Axon.dense(32)
      |> Axon.relu()
      |> Axon.dense(3)
      |> Axon.softmax()

    model
  end

  def train(model, {train_features, train_labels, test_features, test_labels}) do
    # Convert labels to one-hot encoding
    train_labels_onehot = Nx.equal(train_labels, Nx.tensor([[0], [1], [2]]))

    # Configure training parameters
    optimizer = Axon.Optimizers.adam(learning_rate: 0.001)

    # Train the model
    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, optimizer)
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(
      Stream.repeatedly(fn -> {train_features, train_labels_onehot} end),
      %{},
      epochs: 100,
      iterations: div(elem(Nx.shape(train_features), 0), 32)
    )
  end

  def evaluate(model, trained_model_state, {_train_features, _train_labels, test_features, test_labels}) do
    # Convert test labels to one-hot encoding
    test_labels_onehot = Nx.equal(test_labels, Nx.tensor([[0], [1], [2]]))

    # Evaluate the model
    metrics =
      model
      |> Axon.Loop.evaluator()
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.run(test_features, test_labels_onehot, trained_model_state)

    metrics
  end

  def predict(model, trained_model_state, features) do
    Axon.predict(model, trained_model_state, features)
  end
end
