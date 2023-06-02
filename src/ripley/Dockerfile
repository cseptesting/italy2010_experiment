FROM rocker/r-base:4.0.4

# Set group and user
ARG USERNAME=italy
ARG USER_UID=1100
ARG USER_GID=$USER_UID
RUN groupadd -o -g $USER_GID $USERNAME \
    && useradd -o -u $USER_UID -g $USER_GID -s /bin/sh -m $USERNAME


# Set path, env and requirements
WORKDIR /usr/src/kripley/
ENV CONDA_PATH=/usr/mambaforge
ENV PATH="$CONDA_PATH/bin:${PATH}"
ENV CONDA_ENV=kripley
COPY --chown=$USER_UID:$USER_GID environment.yaml ./

# Get basic OS libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget &&  \
    rm -rf /var/lib/apt/lists/*

# Set up python (conda) environment
RUN wget https://github.com/conda-forge/miniforge/releases/download/23.1.0-1/Mambaforge-23.1.0-1-Linux-x86_64.sh && \
    bash Mambaforge-23.1.0-1-Linux-x86_64.sh -b -p $CONDA_PATH && \
    rm -f Mambaforge-23.1.0-1-Linux-x86_64.sh && \
    echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda env create -f environment.yaml && \
    conda activate $CONDA_ENV

# Set up R dependencies
RUN Rscript -e "nproc <- as.integer(commandArgs(trailingOnly = TRUE)[1]); \
                install.packages('remotes', Ncpus=nproc) ; \
                require(remotes); \
                install_version('spatstat.core', version = '2.2-0', Ncpus=nproc); \
                install_version('spatstat.geom', version = '2.2-0', Ncpus=nproc)" $(nproc)

# Unzip models into the same relative path as Local repo
ADD --chown=$USER_UID:$USER_GID models.tar.gz ../../
RUN chown -R $USER_UID:$USER_GID /usr/models/

## Running as user
USER $USERNAME
RUN conda init bash && . /home/$USERNAME/.bashrc

# Entrypoint runs the code immediately
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "kripley",\
            "python", "main.py"]

