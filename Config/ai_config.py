from pathlib import Path

from omegaconf import DictConfig, OmegaConf


class AiConfig:
    """Load the shared configuration and expose its sections."""

    DEFAULT_PATH = Path(__file__).resolve().with_name("config.yaml")

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            self.DEFAULT_PATH if path is None else Path(path).expanduser().resolve()
        )
        loaded = OmegaConf.load(self.path)

        if not isinstance(loaded, DictConfig):
            raise TypeError(
                f"Expected a configuration mapping in {self.path}, "
                f"got {type(loaded).__name__}"
            )

        for name in ("data", "train", "model", "tokenization"):
            section = loaded.get(name)
            if not isinstance(section, DictConfig):
                raise TypeError(
                    f"Expected configuration section '{name}' to be a mapping "
                    f"in {self.path}, got {type(section).__name__}"
                )

        self._config = loaded

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def data_cnf(self) -> DictConfig:
        return self._config["data"]

    @property
    def train_cnf(self) -> DictConfig:
        return self._config["train"]

    @property
    def model_cnf(self) -> DictConfig:
        return self._config["model"]

    @property
    def tokenization_cnf(self) -> DictConfig:
        return self._config["tokenization"]


config = AiConfig()
