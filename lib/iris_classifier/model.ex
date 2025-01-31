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
    # Convert labels to one-hot encoding - fixed version
    train_labels_onehot =
      train_labels
      # Remove extra dimension
      |> Nx.squeeze()
      # Add back the axis at the end
      |> Nx.new_axis(-1)
      # Create one-hot encoding
      |> Nx.equal(Nx.tensor([0, 1, 2]))

    IO.inspect(Nx.shape(train_labels_onehot), label: "One-hot encoded labels shape")

    # Configure training parameters
    optimizer = Polaris.Optimizers.adam(learning_rate: 0.001)

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

  def evaluate(
        model,
        trained_model_state,
        {_train_features, _train_labels, test_features, test_labels}
      ) do
    # Convert test labels to one-hot encoding - fixed version
    test_labels_onehot =
      test_labels
      # Remove extra dimension
      |> Nx.squeeze()
      # Add back the axis at the end
      |> Nx.new_axis(-1)
      # Create one-hot encoding
      |> Nx.equal(Nx.tensor([0, 1, 2]))

    # Create a single batch of test data
    test_data = {test_features, test_labels_onehot}

    # Evaluate the model
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(
      Stream.repeatedly(fn -> test_data end),
      trained_model_state,
      # Just one iteration for the whole test set
      iterations: 1
    )
  end

  def predict(model, trained_model_state, features) do
    Axon.predict(model, trained_model_state, features)
  end

  def print_predictions(predictions, actual_labels, num_samples \\ 10) do
    predicted_classes =
      predictions
      |> Nx.argmax(axis: 1)
      |> Nx.to_flat_list()

    actual_classes =
      actual_labels
      |> Nx.squeeze()
      |> Nx.to_flat_list()

    # Get probabilities for each class
    probabilities =
      Nx.to_flat_list(predictions)
      |> Enum.chunk_every(3)

    IO.puts("\nDetailed predictions:")

    Enum.zip([predicted_classes, actual_classes, probabilities])
    |> Enum.take(num_samples)
    |> Enum.each(fn {pred, actual, probs} ->
      {setosa, versicolor, virginica} = List.to_tuple(probs)
      pred_name = class_to_name(pred)
      actual_name = class_to_name(actual)

      IO.puts("""
      Predicted: #{pred_name} (#{pred}), Actual: #{actual_name} (#{actual})
      Probabilities:
        Setosa: #{Float.round(setosa * 100, 2)}%
        Versicolor: #{Float.round(versicolor * 100, 2)}%
        Virginica: #{Float.round(virginica * 100, 2)}%
      """)
    end)
  end

  defp class_to_name(0), do: "Iris-setosa"
  defp class_to_name(1), do: "Iris-versicolor"
  defp class_to_name(2), do: "Iris-virginica"
end
