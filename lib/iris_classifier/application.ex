defmodule IrisClassifier.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      IrisClassifierWeb.Telemetry,
      IrisClassifier.Repo,
      {DNSCluster, query: Application.get_env(:iris_classifier, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: IrisClassifier.PubSub},
      # Start the Finch HTTP client for sending emails
      {Finch, name: IrisClassifier.Finch},
      # Start a worker by calling: IrisClassifier.Worker.start_link(arg)
      # {IrisClassifier.Worker, arg},
      # Start to serve requests, typically the last entry
      IrisClassifierWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: IrisClassifier.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    IrisClassifierWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
