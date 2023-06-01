FROM rocker/r-base:4.0.4

ARG USERNAME=italy
ARG USER_UID=1100
ARG USER_GID=$USER_UID
RUN groupadd --non-unique -g $USER_GID $USERNAME \
    && useradd -u $USER_UID -g $USER_GID -s /bin/sh -m $USERNAME


WORKDIR /usr/src/kripley/


RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget &&  \
    rm -rf /var/lib/apt/lists/*

COPY --chown=$USER_UID:$USER_GID extra_reqs.txt ./

ENV CONDA_PATH=/usr/miniconda3
ENV PATH="$CONDA_PATH/bin:${PATH}"
ENV CONDA_ENV=kripley


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_PATH\
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n $CONDA_ENV && \
    conda activate $CONDA_ENV && \
    conda install python=3.10 pip &&  \
    conda install -c conda-forge pycsep -y &&\
    pip install -r extra_reqs.txt && \
    git clone https://github.com/cseptesting/floatcsep.git  --branch=main  \
    --depth=1 && pip install floatcsep/


RUN Rscript -e "nproc <- as.integer(commandArgs(trailingOnly = TRUE)[1]); \
                install.packages('remotes', Ncpus=nproc) ; \
                require(remotes); \
                install_version('spatstat.core', version = '2.2-0', Ncpus=nproc); \
                install_version('spatstat', version = '2.2-0', Ncpus=nproc)" $(nproc)

ADD --chown=$USER_UID:$USER_GID models.tar.gz ../../

USER $USERNAME
RUN conda init bash && . /home/$USERNAME/.bashrc

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "kripley",\
            "python", "k_function.py"]

