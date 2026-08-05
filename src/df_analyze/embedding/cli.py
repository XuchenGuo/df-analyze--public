from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from argparse import ArgumentParser, RawTextHelpFormatter
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type, Union

from df_analyze.cli.parsing import (
    resolved_path,
)
from df_analyze.runtime.hardware import DeviceIntent, RuntimePolicy, get_runtime
from df_analyze.runtime.install import DeviceInstall

if TYPE_CHECKING:
    pass
from df_analyze.utils import Debug

USAGE_STRING = """

The df-embed script can be used to embed image (vision) data, or texts (NLP)
which are not too long.

IMPORTANT: In order to get this working, you will first have to download the
underlying HuggingFace zero-shot models. Assuming you have used `cd` to
change to where you have cloned this repo, then this can be done by running:

    python df-embed.py --modality <nlp|vision> --download

*once*. This will download the necessary files to ./downloaded_models. Then,
all subsequent calls will work properly, even offline without internet
access.

Good defaults are chosen so that likely the only arguments you would wish to
specify manually are:

    --data (required)
    --modality (required)

"""

USAGE_EXAMPLES = """
USAGE EXAMPLE:

Assuming you have run:

    `python df-embed --download --modality <nlp|vision>`

successfully, then basic usage is:

    python df-embed.py --data my_data.parquet --modality <nlp|vision> --out ./my_data_embedded.parquet

"""

DATA_HELP = """
Path to the input .parquet data file to be converted to tabular format via
zero-shot embeddings.

# IMAGE DATA

  For image classification data (`python df-embed.py --modality vision`), the
  file must be a two-column table with the columns named "image" and "label".
  The order of the columns is not important, but the "label" column must
  contain integers in {0, 1, ..., c - 1}, where `c` is the number of class
  labels for your data. The data type is not really important, however, if
  the table is loaded into a Pandas DataFrame `df`, then running
  `df["label"].astype(np.int64)` (assuming you have imported NumPy as `np`,
  as is convention) should not alter the meaning of the data.

  For image regression data (very rare), the file must be a two-column table
  with the columns named "image" and "target". The order of the columns is
  not important, but the "target" column must contain floating point values.
  The floating point data type is not really important, however, if the table
  is loaded into a Pandas DataFrame `df`, then running
  `df["label"].astype(float)` should not raise any exceptions.

  The "image" column must be of `bytes` dtype, and must be readable by PIL
  `Image.open`. Internally, all we do, again assuming that the data is loaded
  into a Pandas DataFrame `df`, is run:

      ```python
      from io import BytesIO
      from PIL import Image

      df["image"].apply(lambda raw: Image.open(BytesIO(raw)).convert("RGB"))
      ```

  to convert images to the necessary format. This means that if you load your
  images using PIL `Image.open`, and you have a list of image paths (and a way
  to infer the target from that path, e.g. `get_target(path: Path)`, then you
  can convert your images to bytes through the use of `io` `BytesIO` objects,
  and build your parquet file with just a few lines of Python:

      ```python
      converted: list[bytes] = []
      targets = []

      for path in my_image_paths:
          img = Image.open(path)
          buf = BytesIO()
          img.save(buf, format="JPEG")
          byts = bug.getvalue()
          converted.append(byts)
          targets.append(get_target(path))

      df = DataFrame({"image": converted, "target": targets})
      df.to_parquet("images.parquet")
      ```


# NLP Text Data

  For text classification data (`python df-embed.py --modality nlp`), the
  file must be a two-column table with the columns named "text" and "label".
  The order of the columns is not important, but the "label" column must
  contain integers in {0, 1, ..., c - 1}, where `c` is the number of class
  labels for your data. The data type is not really important, however, if
  the table is loaded into a Pandas DataFrame `df`, then running
  `df["label"].astype(np.int64)` (assuming you have imported NumPy as `np`,
  as is convention) should not alter the meaning of the data.

  For text regression data (e.g. sentiment analysis, rating prediction), the
  file must be a two-column table with the columns named "text" and
  "target". The order of the columns is not important, but the "target"
  column must contain floating point values. The floating point data type is
  not really important, however, if the table is loaded into a Pandas
  DataFrame `df`, then running `df["label"].astype(float)` should not raise
  any exceptions.

  The "text" column will have "object" ("O") dtype. Assuming you have loaded
  your text data into a Pandas DataFrame `df`, then you can check that the
  data has the correct type by running:

      ```python
      assert df.text.apply(lambda s: isinstance(s, str)).all()
      ```

  which will raise an AssertionError if a row has an incorrect type.

  In order to keep compute times reasonable, it is best for text samples
  to be at most a paragraph or two. I.e. the underlying model is not really
  intended for efficient or effective document embedding. However, this
  ultimately depends on the text language and it is hard to make general
  recommendations here.

"""

OUT_HELP = """
Name of the .parquet file in which the embedded data will be saved. Can be
a name, filename and extension, or a directory.

If passing in a name, e.g. 'my_embeddings', then the .parquet extension is
automatically appended. Extensions other than .parquet are not valid and will
raise an exception.

For safety reasons, and to avoid file access issues on e.g. compute clusters,
paths ABOVE the parent of the input data are not permitted. E.g. if your
input image data to be embedded is in:

    /path/to/my/inputs/my_images.parquet

then you will only be able to specify outputs such as:

    /path/to/my/inputs/my_embeddings.parquet
    /path/to/my/inputs/images/my_embeddings.parquet
    /path/to/my/inputs/images/[...]/my_embeddings.parquet

and so forth. If working on an HPC cluster (especially a cluster with
different $HOME and $SCRATCH filesystems, as on e.g. SLURM clusters in
Compute Canada / DRAC, it is highly recommended that you use the `realpath`
or `readlink -f` command first to find out the actual path to your input
file. I.e. running:

    python df-embed.py --data ./my_images.parquet [...]

tends to be a recipe for trouble. You might instead try:

    python df-embed.py --data "$(realpath ./my_images.parquet)" [...]
    python df-embed.py --data "$(readlink -f ./my_images.parquet)" [...]

to avoid these issues.

If you do not specify any output file, then the default is for the output to
go to a file `embeddings.parquet` in the parent directory of the input
parquet file passed to `--data`

"""

NAME_HELP = """
Unique name to label the dataset for debugging / testing purposes. Only
alters CLI outputs and / or produced test files.

"""

FORCE_DL_HELP = """
Download the NLP and vision zero-shot embedding models, overwriting existing
files if present.

"""

LIMIT_HELP = """
If an integer N is passed, only embed the first N samples.

"""

BATCH_HELP = """
Batch size for processing embeddings. Mostly intended for testing, and should
generally be a small integer value like 2, 4, or 8.

"""

DEVICE_HELP = """
Choose where embedding runs. `auto` (recommended) uses CUDA when available and
tries the complete embedding once more on CPU after a CUDA runtime error. `cpu`
disables GPU use. `cuda` requires CUDA and stops on CUDA runtime errors.
"""

DEVICE_INSTALL_HELP = """
Choose whether df-embed may create a separate CUDA-enabled PyTorch environment:
`auto`, `ask`, or `never` (the default).
"""


class EmbeddingModality(Enum):
    NLP = "nlp"
    Vision = "vision"


class EmbeddingOptions(Debug):
    def __init__(
        self,
        datapath: Optional[Path],
        modality: Union[str, EmbeddingModality],
        name: Optional[str] = None,
        outpath: Optional[Path] = None,
        limit_samples: Optional[int] = None,
        batch_size: Optional[int] = 2,
        download: bool = False,
        force_download: bool = False,
        device: Union[str, DeviceIntent] = DeviceIntent.Auto,
        device_install: Union[str, DeviceInstall, None] = None,
    ) -> None:
        # memoization-related
        # other
        any_download = download or force_download
        self.datapath: Optional[Path] = self.validate_datapath(datapath, any_download)
        self.modality = EmbeddingModality(modality)
        if self.datapath is not None:
            self.name = name or self.datapath.stem
            self.outpath = outpath or self.datapath.parent / "embedded.parquet"
        else:
            self.name = None
            self.outpath = None
        self.limit_samples = limit_samples
        self.batch_size = batch_size
        self.download: bool = download
        self.force_download: bool = force_download
        self.any_download = any_download
        self.device = DeviceIntent.from_arg(device)
        self.device_install = DeviceInstall.from_arg(
            device_install, self.device.value
        )

    @property
    def runtime(self) -> RuntimePolicy:
        return get_runtime(self.device)

    @staticmethod
    def validate_datapath(df_path: Optional[Path], any_download: bool) -> Optional[Path]:
        if any_download:
            return
        if df_path is None:
            raise ValueError(
                "No data path (`--data`) was provided, and neither "
                "`--download` nor `--force-download` was specified. If you are not "
                "pre-downloading a model for later use, then you must specifify a "
                "data path with the `--data` argument."
            )
        datapath = resolved_path(df_path)
        if not datapath.exists():
            raise FileNotFoundError(f"The specified file {datapath} does not exist.")
        if not datapath.is_file():
            raise FileNotFoundError(f"{datapath} is not a file.")
        return Path(datapath).resolve()

    @classmethod
    def from_parser(
        cls: Type[EmbeddingOptions], parser: ArgumentParser
    ) -> EmbeddingOptions:
        args = parser.parse_known_args()[0]
        return cls(
            datapath=args.data,
            modality=args.modality,
            name=args.name,
            outpath=args.out,
            limit_samples=args.limit_samples,
            batch_size=args.batch_size,
            download=args.download,
            force_download=args.force_download,
            device=args.device,
            device_install=args.device_install,
        )

    def __str__(self) -> str:
        cls = self.__class__.__name__
        mod = self.modality.name
        fields = {**self.__dict__}
        fields.pop("datapath")
        fields.pop("name")
        fields.pop("modality")
        info = []
        for key, value in fields.items():
            info.append(f"    '{key}': {repr(value)}")  # yes, will break on recursive
        info = "\n".join(info)
        if self.datapath is not None:
            return f"{cls}({self.name}.{mod} @ {self.datapath.parent}\n{info}\n)"
        else:
            if self.name is not None:
                return f"{cls}({self.name}.{mod}@DOWNLOAD-ONLY\n{info}\n)"
            else:
                return f"{cls}({mod}@DOWNLOAD-ONLY\n{info}\n)"

    __repr__ = __str__


def make_parser() -> ArgumentParser:
    def pos_int(s: str) -> int:
        try:
            value = int(s)
            if value < 1:
                raise ValueError("Integer arguments must be positive and nonzero")
            return value
        except Exception as e:
            raise ValueError(f"Invalid positive integer: {s}") from e

    parser = ArgumentParser(
        prog="df-embed",
        usage=USAGE_STRING,
        formatter_class=RawTextHelpFormatter,
        epilog=USAGE_EXAMPLES,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=DATA_HELP,
    )
    parser.add_argument(
        "--modality",
        choices=["vision", "nlp"],
        help="specify the input data type (modality)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        required=False,
        help=NAME_HELP,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=OUT_HELP,
    )
    parser.add_argument(
        "--limit-samples",
        type=pos_int,
        default=None,
        help=LIMIT_HELP,
    )
    parser.add_argument(
        "--batch-size",
        type=pos_int,
        default=2,
        help=BATCH_HELP,
    )
    parser.add_argument(
        "--device",
        type=DeviceIntent.parse,
        choices=DeviceIntent.choices(),
        default=DeviceIntent.Auto.value,
        help=DEVICE_HELP,
    )
    parser.add_argument(
        "--device-install",
        type=DeviceInstall.parse,
        choices=DeviceInstall.choices(),
        default=None,
        help=DEVICE_INSTALL_HELP,
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the NLP and vision zero-shot embedding models.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help=FORCE_DL_HELP,
    )
    # subparsers = parser.add_subparsers(
    #     title="embedding",
    #     description="Tools for embedding image or NLP datasets into a tabular form",
    #     dest="embedding",
    #     required=False,
    #     help="I am subcommand!",
    # )
    return parser
