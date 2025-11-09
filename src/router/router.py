from logging import getLogger

from router.model import VLLMModel, VLLMModelList
from router.utils import HTTP_OK, Request, requests

logger = getLogger("uvicorn")


class Router:
    PORT = 8000
    PORT_RANGE = range(8001, 8011)

    def __init__(self) -> None:
        self.model_map: dict[str, VLLMModel] = {}
        self.refresh()

    @property
    def models(self) -> VLLMModelList:
        return VLLMModelList(data=self.model_map.values())

    @property
    def model_ids(self):
        return [model.root for model in self.models]

    def refresh(self) -> None:
        self.model_map = self._discover()

    def model_exists(self, model_name: str) -> bool:
        return model_name in self.model_map

    def get_model_port(self, model_name: str) -> int:
        return self.model_map[model_name].port

    def _discover(self) -> dict[str, VLLMModel]:
        model_map = {}

        for port in self.PORT_RANGE:
            if not self._get_health(port):
                continue

            try:
                model_data = self._get_model_data(port=port)
                model = VLLMModel(**model_data, port=port)
                model_map[model.id] = model
                logger.info(f"Discovered: {model.id} on port {port}")
            except (TypeError, ValueError) as e:
                logger.warning(f"Port {port}: Invalid model data - {e}")
                continue

        return model_map

    def _get_model_data(self, port: int) -> dict:
        r = Request.model(port=port)
        j = r.json()

        # list of dicts, vllm only has one model-dict
        (model_data,) = j.get("data", {})

        return model_data

    def _get_health(self, port: int) -> bool:
        try:
            r = Request.health(port)
            r.raise_for_status()
            return r.status_code == HTTP_OK
        except requests.RequestException:
            return False
