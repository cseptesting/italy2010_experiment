# Floating Experiment - Italy time independent

## K-ripley analysis

### Docker-Build

```shell
cd src/ripley
docker build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t kripley .
```

### Docker run

```shell
 docker run --rm --volume $PWD:/usr/src/kripley:rw kripley 
```

### Docker interactive run

```shell
 docker run -it --rm --volume $PWD:/usr/src/kripley:rw kripley /bin/bash
conda activate kripley
python k_function.py
```