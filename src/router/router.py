from logging import getLogger

from router.model import VLLMModel
from router.utils import HTTP_OK, Request, requests

logger = getLogger(__name__)


class Router:
    PORT = 8000
    PORT_RANGE = range(8001, 8011)

    def __init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self.model_map = self._discover()

    def _discover(self) -> dict[str, VLLMModel]:
        model_map = {}
        for port in self.PORT_RANGE:
            if self._hearts_alive(port):
                model_data = self._get_model_data(port=port)
                try:
                    model = VLLMModel(**model_data, port=port)
                except (TypeError, ValueError) as e:
                    logger.info(f"port {port}: {e}")
                    continue
                model_map[model.id] = model

        return model_map

    def _get_model_data(self, port: int) -> dict:
        r = Request.model(port=port)
        j = r.json()

        # list of dicts, vllm only has one model-dict
        (model_data,) = j.get("data", {})

        return model_data

    def _get_status(self, port: int):
        r = Request.status(port=port)
        j = r.json()

        return j.get("is_sleeping", False)

    def _hearts_alive(self, port: int) -> bool:
        try:
            r = Request.health(port)
            r.raise_for_status()
            return r.status_code == HTTP_OK
        except requests.RequestException:
            return False

    def model_exists(self, model_name: str) -> bool:
        return model_name in self.model_map

    def get_model_port(self, model_name: str) -> int:
        return self.model_map[model_name].port

    @property
    def models(self):
        return list(self.model_map.values())

    @property
    def model_ids(self):
        return [model.root for model in self.models]
