import requests


class Request:
    @staticmethod
    def health(port: int) -> requests.Response:
        url = f"http://localhost:{port}"
        r = requests.get(url + "/health")
        r.raise_for_status()
        return r

    @staticmethod
    def model(port: int) -> requests.Response:
        url = f"http://localhost:{port}/v1/models"
        r = requests.get(url)
        r.raise_for_status()
        return r

    @staticmethod
    def status(port: int) -> requests.Response:
        url = f"http://localhost:{port}/is_sleeping"
        r = requests.get(url)
        r.raise_for_status()
        return r
