defmodule IrisClassifierWeb.ClassifierLive do
  use IrisClassifierWeb, :live_view

  alias IrisClassifier.Model
  alias IrisClassifier.Dataset

  @impl true
  def mount(_params, _session, socket) do
    # Load the trained model state
    model_state = Model.load_model_state("priv/model_state.etf")

    # Initialize the form with default values
    socket =
      socket
      |> assign(:sepal_length, 5.1)
      |> assign(:sepal_width, 3.5)
      |> assign(:petal_length, 1.4)
      |> assign(:petal_width, 0.2)
      |> assign(:prediction, nil)
      |> assign(:model_state, model_state)

    {:ok, socket}
  end

  @impl true
  def handle_event(
        "predict",
        %{"sepal_length" => sl, "sepal_width" => sw, "petal_length" => pl, "petal_width" => pw},
        socket
      ) do
    # Convert input to tensor
    input =
      Nx.tensor([
        [String.to_float(sl), String.to_float(sw), String.to_float(pl), String.to_float(pw)]
      ])

    # Make prediction
    model = Model.build_model()
    prediction = Model.predict(model, socket.assigns.model_state, input)

    # Extract the predicted class
    predicted_class =
      prediction
      # Get the index of the highest probability
      |> Nx.argmax(axis: 1)
      # Convert tensor to a list
      |> Nx.to_flat_list()
      # Get the first (and only) element
      |> List.first()

    # Map the predicted class index to the class name
    class_name =
      case predicted_class do
        0 -> "Iris-setosa"
        1 -> "Iris-versicolor"
        2 -> "Iris-virginica"
        _ -> "Unknown"
      end

    {:noreply, assign(socket, :prediction, class_name)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div>
      <h1>Iris Flower Classification</h1>
      <form phx-submit="predict">
        <label>Sepal Length:</label>
        <input type="number" step="0.1" name="sepal_length" value={@sepal_length} required />
        <br />
        <label>Sepal Width:</label>
        <input type="number" step="0.1" name="sepal_width" value={@sepal_width} required />
        <br />
        <label>Petal Length:</label>
        <input type="number" step="0.1" name="petal_length" value={@petal_length} required />
        <br />
        <label>Petal Width:</label>
        <input type="number" step="0.1" name="petal_width" value={@petal_width} required />
        <br />
        <button type="submit">Predict</button>
      </form>

      <div :if={@prediction}>
        <h2>Prediction: {@prediction}</h2>
      </div>
    </div>
    """
  end
end
