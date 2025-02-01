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
      |> assign(:current_epoch, 0)
      |> assign(:epochs_history, [])
      |> assign(:show_architecture, false)
      |> assign(:show_training_log, false)
      |> assign(:training_logs, [])

    {:ok, socket, temporary_assigns: [epochs_history: [], training_logs: []]}
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
        send(pid, {:training_log, "Starting dataset load..."})
        {train_features, train_labels, test_features, test_labels} = Dataset.load_iris()
        send(pid, {:training_log, "Dataset loaded successfully"})

        # Build and train model
        send(pid, {:status, "Training model..."})
        send(pid, {:training_log, "Initializing model architecture..."})
        model = Model.build_model()
        send(pid, {:training_log, "Model architecture created"})

        # Train the model
        send(pid, {:training_log, "Beginning training process..."})

        trained_model_state =
          Model.train(model, {train_features, train_labels, test_features, test_labels})

        # Evaluate the model
        send(pid, {:status, "Evaluating model..."})
        send(pid, {:training_log, "Evaluating model performance..."})

        metrics =
          Model.evaluate(
            model,
            trained_model_state,
            {train_features, train_labels, test_features, test_labels}
          )

        # Save the model
        send(pid, {:training_log, "Saving trained model..."})
        Model.save_model_state(trained_model_state, @model_state_path)
        send(pid, {:training_log, "Model saved successfully"})

        # Send completion message
        send(pid, {:training_complete, trained_model_state, metrics})
      rescue
        e ->
          send(pid, {:training_log, "Error occurred: #{Exception.message(e)}"})
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

  def handle_event("toggle_architecture", _, socket) do
    {:noreply, assign(socket, :show_architecture, !socket.assigns.show_architecture)}
  end

  def handle_event("toggle_training_log", _, socket) do
    {:noreply, assign(socket, :show_training_log, !socket.assigns.show_training_log)}
  end

  @impl true
  def handle_info({:status, status}, socket) do
    {:noreply, assign(socket, :training_status, status)}
  end

  def handle_info({:training_metrics, epoch, metrics}, socket) do
    epoch_data = %{
      epoch: epoch,
      accuracy: get_in(metrics, [0, "accuracy"]) |> Nx.to_number(),
      loss: get_in(metrics, [0, "loss"]) |> Nx.to_number()
    }

    socket =
      socket
      |> assign(:current_epoch, epoch)
      |> assign(:epochs_history, [epoch_data | socket.assigns.epochs_history])

    # Broadcast the updated metrics to the chart
    send_update(socket.assigns.myself, metrics: epoch_data)

    {:noreply, socket}
  end

  def handle_info({:training_log, message}, socket) do
    timestamp = DateTime.utc_now() |> DateTime.to_string()
    log_entry = "#{timestamp}: #{message}"

    socket = update(socket, :training_logs, fn logs -> [log_entry | logs] end)
    {:noreply, socket}
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
end
