"""The control API: a REST interface (with Swagger) to a training run, served from the trainer's own process.

``hub`` is what the API and the training loop share, ``keys`` plugs it into the loop's key handling, ``app`` is the
FastAPI application and ``server`` runs it in a thread. See docs/control-api.md."""
