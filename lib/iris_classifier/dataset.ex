defmodule IrisClassifier.Dataset do
  alias Explorer.DataFrame
  alias Explorer.Series

  @iris_csv_path "priv/iris.csv"

  def load_iris() do
    # Load the dataset from the CSV file with headers
    IO.puts("Loading CSV...")
    df = DataFrame.from_csv!(@iris_csv_path)
    IO.inspect(DataFrame.head(df), label: "First few rows")

    IO.puts("Extracting features...")

    # Get feature lists
    sepal_length = df |> DataFrame.pull("sepal_length") |> Series.to_list()
    sepal_width = df |> DataFrame.pull("sepal_width") |> Series.to_list()
    petal_length = df |> DataFrame.pull("petal_length") |> Series.to_list()
    petal_width = df |> DataFrame.pull("petal_width") |> Series.to_list()

    # Create feature vectors and validate them
    feature_vectors =
      0..(length(sepal_length) - 1)
      |> Enum.map(fn i ->
        feature_vector = [
          Enum.at(sepal_length, i),
          Enum.at(sepal_width, i),
          Enum.at(petal_length, i),
          Enum.at(petal_width, i)
        ]

        if Enum.any?(feature_vector, &is_nil/1) do
          IO.puts("Warning: Found nil value at index #{i}")
          IO.inspect(feature_vector, label: "Vector with nil")
        end

        feature_vector
      end)
      |> Enum.reject(fn vector ->
        Enum.any?(vector, &is_nil/1) || Enum.any?(vector, &(not is_number(&1)))
      end)

    IO.puts("Total valid feature vectors: #{length(feature_vectors)}")
    IO.inspect(Enum.take(feature_vectors, 5), label: "First 5 feature vectors")

    # Create features tensor
    features = Nx.tensor(feature_vectors)
    IO.inspect(features, label: "Features tensor shape")

    # Extract and encode labels, matching the number of valid features
    labels = df
    |> DataFrame.pull("species")
    |> Series.to_list()
    |> Enum.take(length(feature_vectors))
    |> Enum.map(&species_to_label/1)
    |> Nx.tensor()
    |> Nx.new_axis(-1)

    IO.inspect(labels, label: "Labels tensor shape")

    # Split into training and testing sets (80% train, 20% test)
    num_samples = Nx.axis_size(features, 0)
    num_train = floor(num_samples * 0.8) |> trunc()

    # Split tensors using tensor indexing
    train_features = Nx.slice(features, [0, 0], [num_train, 4])
    test_features = Nx.slice(features, [num_train, 0], [num_samples - num_train, 4])

    train_labels = Nx.slice(labels, [0, 0], [num_train, 1])
    test_labels = Nx.slice(labels, [num_train, 0], [num_samples - num_train, 1])

    # Normalize features
    {normalized_train_features, mean, std} = normalize_features(train_features)
    normalized_test_features = normalize_features_with_params(test_features, mean, std)

    {normalized_train_features, train_labels, normalized_test_features, test_labels}
  end

  # Helper function to convert species names to numerical labels
  defp species_to_label("Iris-setosa"), do: 0
  defp species_to_label("Iris-versicolor"), do: 1
  defp species_to_label("Iris-virginica"), do: 2

  # Feature normalization functions
  defp normalize_features(features) do
    mean = Nx.mean(features, axes: [0])

    # Calculate standard deviation using variance
    squared_diff = Nx.pow(Nx.subtract(features, mean), 2)
    variance = Nx.mean(squared_diff, axes: [0])
    std = Nx.sqrt(variance)

    # Add small epsilon to avoid division by zero
    std = Nx.add(std, 1.0e-8)

    normalized_features = Nx.divide(Nx.subtract(features, mean), std)
    {normalized_features, mean, std}
  end

  defp normalize_features_with_params(features, mean, std) do
    Nx.divide(Nx.subtract(features, mean), std)
  end
end
