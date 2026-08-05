# Windows Installation

For those who can't use the [shell
script](https://github.com/stfxecutables/df-analyze?tab=readme-ov-file#legacy-local-install-by-shell-script)
to install the necessary `df-analyze` dependencies for a local install, this
should be the attempted installation procedure.

**You must use PowerShell for all commands** (not cmd.exe). For Windows 10 or
11, it is recommended users install the new [Windows
Terminal](https://apps.microsoft.com/detail/9n0dx20hk701), and to use this
instead. Once installed, it will appear in the start menu as "Windows
Powershell", and this will be what you want to use with df-analyze and for
all command-line installation instructions below.

## \*\***NEW**\*\* Installation via `uv`

1. If you haven't already, install [Windows Terminal](https://apps.microsoft.com/detail/9n0dx20hk701)
2. Install [git](https://git-scm.com/install/windows)
   - For those new to the command line, I recommend the following settings during the various dialogues during installation:
      - Make sure "Associate .sh files to be run with Bash" is checked
      - Choose "Visual Studio Code" or "Notepad++" as the default Git editor, NOT Vim
      - In "Adjusting your PATH environment", choose the second or third option
      - Use bundled OpenSSH
      - Use native Windows Secure Channel library
      - In "Configuring line ending conversions" dialogue, choose first or second option ("Checkout Windows, commit Unix" or "Checkout as-is, commit Unix", respectively), NOT the third option ("Checkout as-is, commit as-is")
      - Use MinTTY as the terminal emulator
      - Choose "Only ever fast-forward" for the default behaviour of `git pull`
      - Use Git Credential Manager
      - Leave "Configuring extra options" at defaults (just "Enable file system caching" checked)
2. Install `uv` (https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)
   - i.e. run `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`, and then restart your Windows Terminal
   - be sure to read and understand the notes on execution policies in the link above
3. Make sure `uv` is up-to-date and you have a Python version installed compatible with df-analyze:
   ```shell
   uv self update
   uv python install 3.13.11
   ```
4. Choose a location where you want to install df-analyze, and navigate there in the terminal:

   ```shell
   cd ~\Documents
   git clone https://github.com/stfxecutables/df-analyze.git
   cd df-analyze
   uv sync
   uv pip install pytorch_tabular
   ```
5. Test the install is working by viewing the CLI help for df-analyze by running:

   ```shell
   uv run df-analyze.py --help
   ```

If you get an error at this point involving lines like:

```
 × Failed to build `numpy==1.26.4`
```

or
```
The Meson build system

[...]

Project name: NumPy
Project version: 1.2X.X

[...]

hint: This usually indicates a problem with the package or the build environment.
```

just after running the `uv sync` or `uv pip install pytorch_tabular` commands, you'll need
to use the [Legacy install procedure below](#legacy--fallback-installation-via-pyenv-win).


# Legacy / Fallback Installation via `pyenv-win`

## `pyenv-win` Setup

1. Install [`pyenv-win`](https://github.com/pyenv-win/pyenv-win?tab=readme-ov-file#quick-start)
   by doing the following steps, in order:
   1. In PowerShell, run: `Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"`
   2. Reopen PowerShell
   3. Run `pyenv --version` to check if the installation was successful.
   4. Run `pyenv install 3.13.11`
   5. `cd` to the directory where you cloned `df-analyze`
   6. Run `pyenv local 3.13.11`
      - this creates a permanent file `.python-version` in the `df-analyze`
        directory, and whenever you subsequently open a new PowerShell and `cd`
        to this location, ensures that the correct python version is automatically
        used
   7. Restart PowerShell
   8. `cd` to the directory where you cloned `df-analyze`
   9. Run `python --version` and confirm that the output is `Python 3.13.11`

For all future uses of PowerShell, it should now be that case that if you are in the
`df-analyze` directory, then the correct python version is used. You should always
check this before doing anything else by running:

```powershell
python --version
```

first before doing anything. If for some reason the above does not return `Python 3.13.11`,
you can manually activate the correct python version at any location by running

```powershell
pyenv shell 3.13.11
```

## Virtual Environment Creation

Make sure to `cd` to the `df-analyze` directory, and that `python --version` returns `Python 3.13.11`.
Then, run the following command:

```powershell
python -m venv .venv
```

This creates a virtual environment directory `.venv` in the `df-analyze`
directory. Now, we need to activate this virtual environment (i.e. tell
PowerShell to use the Python binary and libraries contained in the local
`.venv` folder rather than the global python 3.12.5 installation files).
Before activating it, allow locally created scripts for the current Windows
user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

It might be a good idea to restart PowerShell after running this command.

### Activating the Virtual Environment

The environment can then be activated by running one of:

```powershell
.\.venv\Scripts\Activate
```

or

```powershell
.\.venv\Scripts\activate.ps1
```

or

```powershell
.\.venv\Scripts\Activate.ps1
```

To confirm this worked, you should be able to run one of:

```powershell
where python
```

or

```powershell
(get-command python).Path
```

You should see something like `df-analyze/.venv/bin/python.exe` somewhere in the output.
If you do, virtual environment creation was a success.

### Installing Libraries to the Virtual Environment

After you have confirmed that you have installed and activated the virtual
environment, first run the following command to update `pip`:

```powershell
python -m pip install --upgrade pip setuptools wheel --no-cache-dir
```

Then run the following command:

```powershell
python -m pip install catboost cli-test-helpers joblib jsonpickle lightgbm xgboost pykan tabpfn llvmlite matplotlib numba numpy openpyxl optuna pandas pyarrow pytest "pytest-xdist[psutil]" python-dateutil scikit-image scikit-learn scipy seaborn statsmodels tabulate torch torchaudio torchvision tqdm typing_extensions skorch "transformers[torch]" accelerate "datasets[vision]" protobuf sentencepiece "pytorch_tabular"
```

All necessary dependencies should now be installed. You can verify that the installation
was a success by running:

```
python df-analyze.py --help
```

This should produce the documentation for the `df-analyze` tabular prediction interface.

## After Installation

Anytime you open PowerShell and navigate to the `df-analyze` directory, you will have
to [activate the virtual environment](#activating-the-virtual-environment) prior to
running `df-analyze`. But you won't have to do any of the other install procedures again.

### TabPFN Authentication and Model Cache

TabPFN downloads licensed model files the first time each selected checkpoint
is used. First accept the corresponding license. Prior Labs recommends its
browser login or an API key from its account page:

```powershell
$env:TABPFN_TOKEN = "<Prior Labs API key>"
```

If TabPFN instead reports a gated Hugging Face repository, accept that
repository's terms and either run `hf auth login` or set a read-only Hugging
Face token in the same terminal:

```powershell
$env:HF_TOKEN = "<Hugging Face read token>"
```

Do not store either token in the repository or commit it to a configuration
file. See the current
[Prior Labs model-access guide](https://docs.priorlabs.ai/how-to-access-gated-models)
for browser, notebook, and offline setup.

If your Windows application-data directory is read-only, or you downloaded the
weights for offline use, choose a cache directory in the PowerShell terminal
where you will run `df-analyze`:

```powershell
$env:TABPFN_MODEL_CACHE_DIR = "$PWD\.df-analyze-runtime\tabpfn-model-cache"
```

Set this variable again in a new terminal, or add it to your Windows environment
variables if you want to keep using the same directory. If the selected path is
not writable, `df-analyze` reports an error before starting the download. When
no path is set and TabPFN's default cache is not writable, `df-analyze` uses a
temporary directory and prints its location.
