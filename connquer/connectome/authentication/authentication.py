import yaml
from pathlib import Path


__all__ = ["NEUPRINT_TOKEN", "CAVE_TOKEN", "FLYWIRE_LOCAL"]


_TOKEN_PATH = Path(__file__).parent / "tokens.yml"


class Authentication:

    name = None

    def __init__(self, path):
        self.path = Path(path)

    def fetch_token(self, name):
        try:
            with open(self.path, "r") as f:
                tokens = yaml.safe_load(f)
                token = tokens[name]
        except KeyError:
            raise ValueError(f"`{name}` must be provided in `{self.path.name}` file!")
        return token

    def __call__(self):
        return self.fetch_token(self.name)


class CaveToken(Authentication):

    name = "cave_token"

    @staticmethod
    def new_cave_token():
        import caveclient
        client = caveclient.CAVEclient()
        client.auth.setup_token(make_new=True)


CAVE_TOKEN = CaveToken(_TOKEN_PATH)
NEUPRINT_TOKEN = type("NeuprintToken", (Authentication,), {"name": "neuprint_token"})(_TOKEN_PATH)
FLYWIRE_LOCAL = type("FlywirePath", (Authentication,), {"name": "flywire_local"})(_TOKEN_PATH)
