# Use Miniconda base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /usr/src/app

# Copy the environment.yml file to the working directory
COPY bayesian_illumination.yml .

# Create the Conda environment
RUN conda env create -f bayesian_illumination.yml

# Ensure the conda environment is activated
SHELL ["conda", "run", "-n", "bayesian-illumination", "/bin/bash", "-c"]

# Copy the rest of the application code to the working directory
COPY . .

# Set the environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1

# Ensure the conda environment is activated
RUN conda run -n bayesian-illumination python -c "import sys; sys.exit(0)"

# Set the entry point to activate the environment and run the application
ENTRYPOINT ["conda", "run", "-n", "bayesian-illumination", "python", "illuminate.py"]
