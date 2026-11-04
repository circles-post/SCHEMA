"""PubMed graph workflow package."""

from .models import ChunkRecord, FullTextRecord, KeywordRecord, PaperRecord, TripleRecord


def main():
    from .workflow import main as run

    return run()


def run_pipeline(*args, **kwargs):
    from .workflow import run_pipeline as run

    return run(*args, **kwargs)
