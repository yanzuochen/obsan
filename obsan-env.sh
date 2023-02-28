# Source me!
# shellcheck disable=SC2148

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "$SCRIPT_DIR" || return

if [ ! -d ../venv ]; then
	if ! which pyenv > /dev/null; then
		eval "$(pyenv init --path)"
		eval "$(pyenv init -)"
	fi

	if ! pyenv shell 3.8.12; then
		pyenv install 3.8.12
		echo "To make sure everything works, restart your shell and re-source this script."
		return
	fi

	echo 'Setting up venv...'
	python3.8 -m venv ../venv
	pyenv shell system
	source ../venv/bin/activate
	echo 'Installing dependencies...'
	python -m pip install torch torchvision torchaudio
	python -m pip install -r requirements.txt
	deactivate
fi

source ../venv/bin/activate

export PYTHONPATH="$SCRIPT_DIR/../python"
