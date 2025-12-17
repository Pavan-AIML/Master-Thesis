<h1 align="center">🎓 Master's Thesis</h1>

<h3 align = "center"> Deep learning focused </h3>


<h4 align="center"> Topic:  Neural Process Models for Spatio-Temporal Reconstruction of Aerosol Optical Depth and PM₂.₅ 🌎 </h4>

<h3 align="center"> Followed steps in "main branch" </h3>


## Before Production (Research & Development Phase)

This stage focuses on data understanding, model design, experimentation, and validation.

Problem Definition & Data Understanding

Define the goal: reconstruct spatial–temporal patterns of AOD and PM₂.₅.

Explore satellite-based and ground-based air quality datasets.

Identify missing data patterns and temporal gaps.

Data Preprocessing & Analysis

Handle missing values, normalization, and scaling.

Conduct exploratory analysis in notebooks/PM2.5_data_loader_analysis.py.

Spatially align AOD and PM₂.₅ readings using geolocation metadata.

## Data Loader Development

Implement a PyTorch-compatible loader (torch_data_loader.py) for efficient batching.

Integrate spatio-temporal sampling strategies for model input consistency.

Spatial Encoding / Positional Embedding

Incorporate spatial coordinates into an embedding space using positional encodings.

Encode latitude–longitude relations to preserve spatial structure in the neural process model.

## Model Architecture Development

Design the Neural Process (NP) architecture inside src/model/.

Experiment with conditional neural processes (CNPs), attentive NPs, or GNPs for spatio-temporal interpolation.

Define the encoder–decoder structure and latent variable formulation.

## Model Training

Train the model with various loss functions (e.g., MSE, KL divergence for variational components).

Use GPU-accelerated training with early stopping and checkpointing.

## Model Evaluation

Validate performance using RMSE, and spatial–temporal consistency metrics.


Unit Testing

Perform functional and integration testing with Pytest to ensure code reliability.

## After Production (Deployment & Operational Phase)

This stage ensures reproducibility, scalability, and model serving in real environments.

Dockerization

Create a Dockerfile containing the environment (Python, PyTorch, dependencies).

Ensure portability across systems (local GPU → cloud instance).

Model Serving / API Deployment

Package the trained model into a RESTful API (Fast API).

Containerize and deploy on a cloud platform or server environment.

Integrate monitoring for inference latency and data drift.

Continuous Integration / Continuous Deployment (CI/CD)

Automate testing and deployment using GitHub Actions or similar pipelines.

Maintain version control for model updates and retraining.

Post-Deployment Monitoring

Track prediction quality, data distribution changes, and model drift.

Schedule retraining with new data periodically to maintain accuracy.


