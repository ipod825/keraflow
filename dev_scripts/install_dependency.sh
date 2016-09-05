#!/bin/bash

py2="2.7"
py3="3.5"
conda2="miniconda2-latest"
conda3="miniconda3-latest"
env2="k2"
env3="k3"
tf_version=0.10
tf_cpu_host=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow
tf_gpu_host=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow
tf_host=$tf_cpu_host

function install_tensorflow(){
    pyn=${1/\./}
    if [[ $1 == $py2 ]]
    then
        pip install $tf_host-$tf_version.0rc0-cp"$pyn"-none-linux_x86_64.whl
    else
        pip install $tf_host-$tf_version.0rc0-cp"$pyn"-cp"$pyn"m-linux_x86_64.whl
    fi
}

function run_version(){
    echo "Setting up environment $2"
    # Install miniconda
    export PYENV_VERSION=system
    pyenv install $1 -s

    # create virtual env
    export PYENV_VERSION=$1
    #pyenv activate $1
    conda update -q conda
    pyenv virtualenv $1 $2

    # building dependencies in created virtual env
    export PYENV_VERSION=$2
    pip install --upgrade -r dev_scripts/dev_requirements.txt
    conda config --set always_yes yes --set changeps1 no
    conda install -q numpy scipy && \
    pip install theano --upgrade && \
    install_tensorflow $3
}

if ! [[ -d $HOME/.pyenv ]]; then
    curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
fi

# Travis use old version without miniconda-latest. Use pre-download pyenv instead.
if [[ "`pyenv --version`" < "pyenv 20160726" ]]; then
    tar -zxf dev_scripts/pyenv20160726.tgz
    cp -r pyenv/* "$HOME/.pyenv"
fi

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

if [[ $TRAVIS_PYTHON_VERSION == $py2 ]]; then
    run_version $conda2 $env2 $py2
elif [[ $TRAVIS_PYTHON_VERSION == $py3 ]]; then
    run_version $conda3 $env3 $py3
else
    echo "Do you want to install tensorflow with gpu enabled? [Enter 1 or 2]"
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) tf_host=$tf_gpu_host
                break;;
            No ) tf_host=$tf_cpu_host
                break;;
        esac
    done
    run_version $conda2 $env2 $py2 && \
    run_version $conda3 $env3 $py3
fi
