# Experiment records and artifacts

## Git stores

- experiment plan;
- experiment tracker;
- Git commit and configuration;
- compact aggregate and per-query results needed to understand the conclusion;
- interpretation, limitations, and next decision.

## Shared artifact storage stores

- datasets;
- feature caches;
- checkpoints;
- model weights;
- full predictions;
- large logs and plots.

## Current storage status

The shared artifact location has not been confirmed. Selector V2 training must not start
until at least two team members can access the same location and one member can read a
checkpoint written by another.

Do not make a private server path the only experiment record.
