"""Question generation components; import the CLI only when invoked."""


def main():
    from .cli import main as run

    return run()

__all__ = ["main"]
