from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


class HistoryStoreConfig(BaseModel):
    provider: str = Field(
        description="Provider of the history store (e.g., 'sqlite', 'mongodb')",
        default="sqlite",
    )
    config: Optional[Dict] = Field(description="Configuration for the specific history store", default=None)

    _provider_configs: Dict[str, str] = {
        "sqlite": "SQLiteConfig",
        "mongodb": "MongoDBConfig",
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "HistoryStoreConfig":
        provider = self.provider
        config = self.config

        if provider not in self._provider_configs:
            raise ValueError(f"Unsupported history store provider: {provider}")

        module = __import__(
            f"mem0.configs.history_stores.{provider}",
            fromlist=[self._provider_configs[provider]],
        )
        config_class = getattr(module, self._provider_configs[provider])

        if config is None:
            config = {}

        if not isinstance(config, dict):
            if not isinstance(config, config_class):
                raise ValueError(f"Invalid config type for provider {provider}")
            return self

        self.config = config_class(**config)
        return self