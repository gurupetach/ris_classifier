defmodule IrisClassifier.Repo do
  use Ecto.Repo,
    otp_app: :iris_classifier,
    adapter: Ecto.Adapters.Postgres
end
