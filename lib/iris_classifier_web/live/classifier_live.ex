defmodule IrisClassifierWeb.ClassifierLive do
  use IrisClassifierWeb, :live_view

  alias IrisClassifier.Model
  alias IrisClassifier.Dataset

  @model_state_path "priv/model_state.etf"

  @impl true
  def mount(_params, _session, socket) do
    # Try to load existing model state
    initial_model_state = load_existing_model_state()
    initial_status = if initial_model_state, do: "Model loaded from saved state", else: nil

    # Initialize with default values and potentially loaded model state
    socket =
      socket
      |> assign(:sepal_length, 5.1)
      |> assign(:sepal_width, 3.5)
      |> assign(:petal_length, 1.4)
      |> assign(:petal_width, 0.2)
      |> assign(:prediction, nil)
      |> assign(:model_state, initial_model_state)
      |> assign(:training_status, initial_status)
      |> assign(:training_metrics, [])
      |> assign(:evaluation_metrics, nil)
      |> assign(:error_message, nil)

    {:ok, socket}
  end

  # Helper function to safely load existing model state
  defp load_existing_model_state do
    if File.exists?(@model_state_path) do
      try do
        Model.load_model_state(@model_state_path)
      rescue
        _ -> nil
      end
    else
      nil
    end
  end

  @impl true
  def handle_event("train_model", _params, socket) do
    # Start training in a separate process to keep the UI responsive
    pid = self()

    Task.start(fn ->
      try do
        # Load dataset
        send(pid, {:status, "Loading dataset..."})
        {train_features, train_labels, test_features, test_labels} = Dataset.load_iris()

        # Build and train model
        send(pid, {:status, "Training model..."})
        model = Model.build_model()

        # Create a function to capture training metrics
        send_metrics = fn epoch, metrics ->
          send(pid, {:training_metrics, epoch, metrics})
        end

        # Train the model with metrics callback
        trained_model_state =
          Model.train(model, {train_features, train_labels, test_features, test_labels})

        # Evaluate the model
        send(pid, {:status, "Evaluating model..."})

        metrics =
          Model.evaluate(
            model,
            trained_model_state,
            {train_features, train_labels, test_features, test_labels}
          )

        # Save the model
        Model.save_model_state(trained_model_state, @model_state_path)

        # Send completion message
        send(pid, {:training_complete, trained_model_state, metrics})
      rescue
        e ->
          send(pid, {:training_error, Exception.message(e)})
      end
    end)

    {:noreply, assign(socket, :training_status, "Starting training...")}
  end

  @impl true
  def handle_event(
        "predict",
        %{"sepal_length" => sl, "sepal_width" => sw, "petal_length" => pl, "petal_width" => pw},
        socket
      ) do
    if socket.assigns.model_state do
      # Convert input to tensor
      input =
        Nx.tensor([
          [String.to_float(sl), String.to_float(sw), String.to_float(pl), String.to_float(pw)]
        ])

      # Normalize input using the same parameters as training data
      # These values should match your training normalization
      means = Nx.tensor([5.843333, 3.057333, 3.758000, 1.199333])
      stds = Nx.tensor([0.828066, 0.435866, 1.765298, 0.762238])

      input = Nx.divide(Nx.subtract(input, means), stds)

      # Make prediction
      model = Model.build_model()
      prediction = Model.predict(model, socket.assigns.model_state, input)

      # Debug logging
      IO.inspect(prediction, label: "Raw prediction tensor")
      IO.inspect(Nx.to_flat_list(prediction), label: "Prediction probabilities")

      # Get probabilities for each class
      probabilities = prediction |> Nx.to_flat_list()

      # Get predicted class
      predicted_class =
        prediction
        |> Nx.argmax(axis: 1)
        |> Nx.to_flat_list()
        |> List.first()

      # Extract individual probabilities
      [setosa_prob, versicolor_prob, virginica_prob] = probabilities

      # Create detailed prediction result
      result = %{
        class: class_name(predicted_class),
        probabilities: %{
          "Iris-setosa" => setosa_prob,
          "Iris-versicolor" => versicolor_prob,
          "Iris-virginica" => virginica_prob
        }
      }

      {:noreply, assign(socket, :prediction, result)}
    else
      {:noreply, assign(socket, :error_message, "Please train the model first")}
    end
  end

  @impl true
  def handle_info({:status, status}, socket) do
    {:noreply, assign(socket, :training_status, status)}
  end

  def handle_info({:training_metrics, epoch, metrics}, socket) do
    updated_metrics = socket.assigns.training_metrics ++ [{epoch, metrics}]
    {:noreply, assign(socket, :training_metrics, updated_metrics)}
  end

  def handle_info({:training_complete, model_state, metrics}, socket) do
    socket =
      socket
      |> assign(:model_state, model_state)
      |> assign(:evaluation_metrics, metrics)
      |> assign(:training_status, "Training complete!")
      |> assign(:error_message, nil)

    {:noreply, socket}
  end

  def handle_info({:training_error, error_message}, socket) do
    socket =
      socket
      |> assign(:training_status, "Training failed")
      |> assign(:error_message, error_message)

    {:noreply, socket}
  end

  defp class_name(0), do: "Iris-setosa"
  defp class_name(1), do: "Iris-versicolor"
  defp class_name(2), do: "Iris-virginica"
  defp class_name(_), do: "Unknown"

  @impl true
  def render(assigns) do
    ~H"""
    <div class="mx-auto  p-6">
      <h1 class="text-3xl font-bold mb-6">Iris Flower Classification</h1>

    <!-- Model Status Banner -->
      <%= if @model_state do %>
        <div class="mb-4 p-3 bg-green-100 text-green-700 rounded flex items-center justify-between">
          <span>Model is loaded and ready for predictions</span>
          <button
            phx-click="train_model"
            class="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600 text-sm"
          >
            Retrain Model
          </button>
        </div>
      <% else %>
        <div class="mb-4 p-3 bg-yellow-100 text-yellow-700 rounded">
          No trained model found. Please train a new model to make predictions.
        </div>
      <% end %>

    <!-- Training Section -->
      <div class="mb-8 p-6 bg-gray-50 rounded-lg">
        <h2 class="text-xl font-semibold mb-4">Model Training</h2>

        <button
          phx-click="train_model"
          disabled={@training_status == "Training..."}
          class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
        >
          Train Model
        </button>

        <%= if @training_status do %>
          <div class="mt-4">
            <p class="font-semibold">Status: {@training_status}</p>
          </div>
        <% end %>

        <%= if @error_message do %>
          <div class="mt-4 p-4 bg-red-100 text-red-700 rounded">
            {@error_message}
          </div>
        <% end %>

        <%= if @evaluation_metrics do %>
          <div class="mt-4">
            <h3 class="font-semibold">Model Metrics:</h3>
            <p>
              Accuracy: {get_in(@evaluation_metrics, [0, "accuracy"])
              |> Nx.to_number()
              |> Float.round(4)}
            </p>
          </div>
        <% end %>
      </div>

    <!-- Prediction Form -->
      <div class="mb-8 p-6 bg-gray-50 rounded-lg">
        <h2 class="text-xl font-semibold mb-4">Make Predictions</h2>

        <form phx-submit="predict" class="space-y-4">
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-sm font-medium mb-1">Sepal Length:</label>
              <input
                type="number"
                step="0.1"
                name="sepal_length"
                value={@sepal_length}
                required
                class="w-full p-2 border rounded"
              />
            </div>

            <div>
              <label class="block text-sm font-medium mb-1">Sepal Width:</label>
              <input
                type="number"
                step="0.1"
                name="sepal_width"
                value={@sepal_width}
                required
                class="w-full p-2 border rounded"
              />
            </div>

            <div>
              <label class="block text-sm font-medium mb-1">Petal Length:</label>
              <input
                type="number"
                step="0.1"
                name="petal_length"
                value={@petal_length}
                required
                class="w-full p-2 border rounded"
              />
            </div>

            <div>
              <label class="block text-sm font-medium mb-1">Petal Width:</label>
              <input
                type="number"
                step="0.1"
                name="petal_width"
                value={@petal_width}
                required
                class="w-full p-2 border rounded"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={is_nil(@model_state)}
            class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:opacity-50"
          >
            Predict
          </button>
        </form>
      </div>

    <!-- Prediction Results -->
      <%= if @prediction do %>
        <div class="p-6 bg-gray-50 rounded-lg">
          <h2 class="text-xl font-semibold mb-4">Prediction Results</h2>

          <div class="mb-4">
            <p class="text-lg">
              Predicted Class: <span class="font-semibold">{@prediction.class}</span>
            </p>
          </div>

          <div>
            <h3 class="font-semibold mb-2">Class Probabilities:</h3>
            <div class="space-y-2">
              <%= for {class, probability} <- @prediction.probabilities do %>
                <div class="flex items-center">
                  <div class="w-32">{class}:</div>
                  <div class="flex-1">
                    <div class="bg-gray-200 h-4 rounded overflow-hidden">
                      <div class="bg-blue-500 h-full" style={"width: #{probability * 100}%"} />
                    </div>
                  </div>
                  <div class="ml-2 w-16 text-right">
                    {Float.round(probability * 100, 1)}%
                  </div>
                </div>
              <% end %>
            </div>
          </div>
        </div>
      <% end %>
    </div>
    """
  end
end
