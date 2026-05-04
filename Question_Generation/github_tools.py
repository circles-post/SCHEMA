"""Standalone GitHub reference tools for experiment generation.

This module is adapted from the STELLA GitHub tools so agents in this
repository can search GitHub for relevant repositories and code examples
before drafting experiment-code questions.

Environment:
- ``GITHUB_TOKEN``: optional but recommended for authenticated GitHub API access

Usage examples:
- ``python github_tools.py repo-search "protein structure prediction" --language Python``
- ``python github_tools.py code-search "calculate_rmsd" --language Python``
- ``python github_tools.py repo-info google jax``
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import requests
from requests.exceptions import RequestException

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from smolagents import tool
except ImportError:  # pragma: no cover - keeps the module usable without smolagents
    def tool(fn):
        fn.forward = fn
        return fn


def _load_local_env() -> None:
    """Load key=value pairs from a local .env file without requiring python-dotenv."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
else:
    _load_local_env()


GITHUB_API_BASE = "https://api.github.com"
DEFAULT_TIMEOUT = 20


def _get_github_headers() -> Dict[str, str]:
    """Build GitHub API headers, using GITHUB_TOKEN when available."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "question-generation-github-tools",
    }
    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def _github_get(url: str, params: Optional[dict] = None) -> requests.Response:
    """Send a GET request to the GitHub API with common headers."""
    return requests.get(url, params=params, headers=_get_github_headers(), timeout=DEFAULT_TIMEOUT)


@tool
def search_github_repositories(
    query: str,
    language: str = "",
    sort: str = "stars",
    order: str = "desc",
    per_page: int = 10,
) -> str:
    """Search GitHub repositories that can be used as coding references.

    Args:
        query: Free-text repository search query.
        language: Optional language filter such as ``Python``.
        sort: Repository sort field accepted by the GitHub search API.
        order: Sort order, typically ``desc`` or ``asc``.
        per_page: Maximum number of repositories to return.

    Returns:
        A formatted plain-text summary of matching repositories.
    """
    try:
        search_query = query
        if language:
            search_query += f" language:{language}"

        response = _github_get(
            f"{GITHUB_API_BASE}/search/repositories",
            params={
                "q": search_query,
                "sort": sort,
                "order": order,
                "per_page": min(per_page, 100),
            },
        )
        response.raise_for_status()

        data = response.json()
        repositories = data.get("items", [])
        if not repositories:
            return f"No repositories found for query: {query}"

        result = f"GitHub repository search results for '{query}':\n\n"
        for i, repo in enumerate(repositories, 1):
            result += f"{i}. {repo.get('full_name', 'N/A')}\n"
            result += f"   Description: {repo.get('description', 'No description available')}\n"
            result += (
                f"   Language: {repo.get('language', 'N/A')} | "
                f"Stars: {repo.get('stargazers_count', 0)} | "
                f"Forks: {repo.get('forks_count', 0)}\n"
            )
            result += f"   Updated: {repo.get('updated_at', 'N/A')[:10]}\n"
            result += f"   URL: {repo.get('html_url', 'N/A')}\n\n"

        result += f"Total repositories found: {data.get('total_count', 0)}"
        return result
    except RequestException as exc:
        return f"GitHub repository search failed: {exc}"
    except Exception as exc:  # pragma: no cover - defensive path
        return f"Unexpected error while searching repositories: {exc}"


@tool
def search_github_code(
    query: str,
    language: str = "",
    filename: str = "",
    extension: str = "",
    per_page: int = 10,
) -> str:
    """Search for code snippets in GitHub repositories.

    Args:
        query: Free-text code search query.
        language: Optional language filter such as ``Python``.
        filename: Optional exact filename filter.
        extension: Optional file extension filter.
        per_page: Maximum number of code search results to return.

    Returns:
        A formatted plain-text summary of matching code files.
    """
    try:
        search_query = query
        if language:
            search_query += f" language:{language}"
        if filename:
            search_query += f" filename:{filename}"
        if extension:
            search_query += f" extension:{extension}"

        response = _github_get(
            f"{GITHUB_API_BASE}/search/code",
            params={
                "q": search_query,
                "per_page": min(per_page, 100),
            },
        )
        response.raise_for_status()

        data = response.json()
        code_items = data.get("items", [])
        if not code_items:
            return f"No code results found for query: {query}"

        result = f"GitHub code search results for '{query}':\n\n"
        for i, item in enumerate(code_items, 1):
            result += f"{i}. {item.get('name', 'N/A')}\n"
            result += f"   Repository: {item.get('repository', {}).get('full_name', 'N/A')}\n"
            result += f"   Path: {item.get('path', 'N/A')}\n"
            result += f"   URL: {item.get('html_url', 'N/A')}\n\n"

        result += f"Total code results found: {data.get('total_count', 0)}"
        return result
    except RequestException as exc:
        return f"GitHub code search failed: {exc}"
    except Exception as exc:  # pragma: no cover - defensive path
        return f"Unexpected error while searching code: {exc}"


def search_github_code_structured(
    query: str,
    language: str = "",
    per_page: int = 5,
) -> dict:
    """Structured variant of ``search_github_code`` for programmatic callers.

    Returns::

        {
            "query":      str,
            "items":      list[dict],   # each: owner, repo, path, html_url, sha
            "total_count": int,
            "error":      str,
        }

    The string-formatting flavour (``search_github_code``) stays untouched —
    use this when downstream code wants to immediately fetch the matching
    files (via ``fetch_github_file``) instead of pasting search results into
    a prompt.
    """
    try:
        search_query = query
        if language:
            search_query += f" language:{language}"
        response = _github_get(
            f"{GITHUB_API_BASE}/search/code",
            params={"q": search_query, "per_page": min(per_page, 100)},
        )
        response.raise_for_status()
        data = response.json()
        items: list[dict] = []
        for raw in data.get("items", []) or []:
            repo = raw.get("repository", {}) or {}
            owner = (repo.get("owner") or {}).get("login", "")
            items.append(
                {
                    "owner": owner,
                    "repo": repo.get("name", ""),
                    "full_name": repo.get("full_name", ""),
                    "path": raw.get("path", ""),
                    "name": raw.get("name", ""),
                    "html_url": raw.get("html_url", ""),
                    "sha": raw.get("sha", ""),
                    "default_branch": repo.get("default_branch", ""),
                }
            )
        return {
            "query": query,
            "items": items,
            "total_count": int(data.get("total_count", 0) or 0),
            "error": "",
        }
    except RequestException as exc:
        return {"query": query, "items": [], "total_count": 0, "error": f"GitHub code search failed: {exc}"}
    except Exception as exc:  # pragma: no cover - defensive path
        return {"query": query, "items": [], "total_count": 0, "error": f"Unexpected error: {exc}"}


@tool
def get_github_repository_info(repo_owner: str, repo_name: str) -> str:
    """Get detailed information about a specific GitHub repository.

    Args:
        repo_owner: Repository owner or organization name.
        repo_name: Repository name.

    Returns:
        A formatted plain-text summary of repository metadata and README preview.
    """
    try:
        repo_response = _github_get(f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}")
        repo_response.raise_for_status()
        repo_data = repo_response.json()

        readme_content = "README unavailable"
        try:
            readme_response = _github_get(f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}/readme")
            if readme_response.ok:
                readme_data = readme_response.json()
                download_url = readme_data.get("download_url")
                if download_url:
                    readme_content = requests.get(
                        download_url,
                        headers=_get_github_headers(),
                        timeout=DEFAULT_TIMEOUT,
                    ).text
        except Exception:
            pass

        release_info = "No release information available"
        try:
            latest_release = _github_get(
                f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}/releases/latest"
            ).json()
            if "tag_name" in latest_release:
                release_info = (
                    f"Latest release: {latest_release.get('tag_name', 'N/A')} "
                    f"({latest_release.get('published_at', 'N/A')[:10]})"
                )
        except Exception:
            pass

        result = f"GitHub repository details: {repo_owner}/{repo_name}\n\n"
        result += f"Description: {repo_data.get('description', 'N/A')}\n"
        result += f"Language: {repo_data.get('language', 'N/A')}\n"
        result += f"Stars: {repo_data.get('stargazers_count', 0)}\n"
        result += f"Forks: {repo_data.get('forks_count', 0)}\n"
        result += f"Size: {repo_data.get('size', 0)} KB\n"
        result += f"Created: {repo_data.get('created_at', 'N/A')[:10]}\n"
        result += f"Updated: {repo_data.get('updated_at', 'N/A')[:10]}\n"
        result += f"{release_info}\n"
        result += f"URL: {repo_data.get('html_url', 'N/A')}\n"
        result += f"Clone URL: {repo_data.get('clone_url', 'N/A')}\n\n"

        topics = repo_data.get("topics", [])
        if topics:
            result += f"Topics: {', '.join(topics)}\n\n"

        if readme_content != "README unavailable":
            result += "README preview:\n"
            result += "=" * 50 + "\n"
            result += readme_content[:1000]
            if len(readme_content) > 1000:
                result += "\n... (truncated to first 1000 characters)"
            result += "\n" + "=" * 50 + "\n"

        return result
    except RequestException as exc:
        return f"Failed to fetch repository details: {exc}"
    except Exception as exc:  # pragma: no cover - defensive path
        return f"Unexpected error while fetching repository details: {exc}"


@tool
def list_github_repository_tree(
    repo_owner: str,
    repo_name: str,
    ref: str = "",
    path_filter: str = "",
    max_entries: int = 200,
) -> dict:
    """Recursively list files in a GitHub repository via the Git Trees API.

    Args:
        repo_owner: Repository owner or organization name.
        repo_name: Repository name.
        ref: Optional branch name or commit SHA. Empty string falls back to
            the repository's default branch.
        path_filter: Optional case-insensitive substring; only entries whose
            path contains this string are kept. For example ``.py`` keeps
            only Python files.
        max_entries: Maximum number of entries to return after filtering.

    Returns:
        A dict with keys ``owner``, ``repo``, ``ref``, ``default_branch``,
        ``truncated`` (bool, True iff GitHub truncated the tree on its end),
        ``entries`` (list of {path, type, size, sha}), and ``error`` (empty
        string on success, otherwise a human-readable failure reason).
    """
    try:
        # Resolve default_branch if no ref was given.
        repo_response = _github_get(f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}")
        repo_response.raise_for_status()
        repo_data = repo_response.json()
        default_branch = str(repo_data.get("default_branch") or "main")
        target_ref = ref or default_branch

        tree_response = _github_get(
            f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}/git/trees/{target_ref}",
            params={"recursive": "1"},
        )
        tree_response.raise_for_status()
        tree_data = tree_response.json()

        raw_tree = tree_data.get("tree", []) or []
        needle = path_filter.casefold() if path_filter else ""
        entries: list[dict] = []
        for node in raw_tree:
            path = node.get("path", "")
            if needle and needle not in path.casefold():
                continue
            entries.append(
                {
                    "path": path,
                    "type": node.get("type", ""),
                    "size": int(node.get("size") or 0),
                    "sha": node.get("sha", ""),
                }
            )
            if len(entries) >= max_entries:
                break

        return {
            "owner": repo_owner,
            "repo": repo_name,
            "ref": target_ref,
            "default_branch": default_branch,
            "truncated": bool(tree_data.get("truncated", False)),
            "entries": entries,
            "error": "",
        }
    except RequestException as exc:
        return {
            "owner": repo_owner,
            "repo": repo_name,
            "ref": ref,
            "default_branch": "",
            "truncated": False,
            "entries": [],
            "error": f"GitHub tree fetch failed: {exc}",
        }
    except Exception as exc:  # pragma: no cover - defensive path
        return {
            "owner": repo_owner,
            "repo": repo_name,
            "ref": ref,
            "default_branch": "",
            "truncated": False,
            "entries": [],
            "error": f"Unexpected error while listing repository tree: {exc}",
        }


@tool
def fetch_github_file(
    repo_owner: str,
    repo_name: str,
    path: str,
    ref: str = "",
    max_bytes: int = 200_000,
) -> dict:
    """Download a single file from a GitHub repository via the Contents API.

    Args:
        repo_owner: Repository owner or organization name.
        repo_name: Repository name.
        path: Path to the file inside the repository, e.g.
            ``examples/run_analysis_example.py``.
        ref: Optional branch name or commit SHA. Empty string falls back to
            the repository's default branch.
        max_bytes: Maximum number of bytes to keep from the downloaded file.
            Larger files are clipped to this size and the returned dict has
            ``truncated=True`` so callers can warn the model.

    Returns:
        A dict with keys ``owner``, ``repo``, ``path``, ``ref``, ``size``
        (bytes as reported by the API), ``encoding`` (``utf-8`` for decoded
        text, ``base64`` for binary), ``content`` (decoded text or base64
        string), ``truncated`` (bool), ``html_url``, ``download_url``, and
        ``error`` (empty string on success).
    """
    try:
        api_url = f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}/contents/{path}"
        params = {"ref": ref} if ref else None
        response = _github_get(api_url, params=params)
        response.raise_for_status()
        meta = response.json()
        if isinstance(meta, list):
            return {
                "owner": repo_owner, "repo": repo_name, "path": path, "ref": ref,
                "size": 0, "encoding": "", "content": "", "truncated": False,
                "html_url": "", "download_url": "",
                "error": "path is a directory, not a file",
            }
        size = int(meta.get("size") or 0)
        download_url = meta.get("download_url") or ""
        html_url = meta.get("html_url") or ""

        # GitHub returns base64 inline for files <1MB; for bigger files the
        # `content` field is empty and we must follow `download_url` instead.
        encoded = meta.get("content") or ""
        if encoded:
            import base64 as _b64
            try:
                raw = _b64.b64decode(encoded)
            except Exception as exc:
                return {
                    "owner": repo_owner, "repo": repo_name, "path": path, "ref": ref,
                    "size": size, "encoding": "", "content": "", "truncated": False,
                    "html_url": html_url, "download_url": download_url,
                    "error": f"failed to base64-decode response: {exc}",
                }
        elif download_url:
            raw_response = requests.get(
                download_url, headers=_get_github_headers(), timeout=DEFAULT_TIMEOUT
            )
            raw_response.raise_for_status()
            raw = raw_response.content
        else:
            return {
                "owner": repo_owner, "repo": repo_name, "path": path, "ref": ref,
                "size": size, "encoding": "", "content": "", "truncated": False,
                "html_url": html_url, "download_url": download_url,
                "error": "no inline content and no download_url",
            }

        truncated = len(raw) > max_bytes
        if truncated:
            raw = raw[:max_bytes]

        try:
            content_text = raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            import base64 as _b64
            content_text = _b64.b64encode(raw).decode("ascii")
            encoding = "base64"

        return {
            "owner": repo_owner,
            "repo": repo_name,
            "path": path,
            "ref": ref or meta.get("sha", ""),
            "size": size,
            "encoding": encoding,
            "content": content_text,
            "truncated": truncated,
            "html_url": html_url,
            "download_url": download_url,
            "error": "",
        }
    except RequestException as exc:
        return {
            "owner": repo_owner, "repo": repo_name, "path": path, "ref": ref,
            "size": 0, "encoding": "", "content": "", "truncated": False,
            "html_url": "", "download_url": "",
            "error": f"GitHub file fetch failed: {exc}",
        }
    except Exception as exc:  # pragma: no cover - defensive path
        return {
            "owner": repo_owner, "repo": repo_name, "path": path, "ref": ref,
            "size": 0, "encoding": "", "content": "", "truncated": False,
            "html_url": "", "download_url": "",
            "error": f"Unexpected error while fetching file: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone GitHub tools for question generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    repo_parser = subparsers.add_parser("repo-search", help="Search GitHub repositories")
    repo_parser.add_argument("query")
    repo_parser.add_argument("--language", default="")
    repo_parser.add_argument("--sort", default="stars")
    repo_parser.add_argument("--order", default="desc")
    repo_parser.add_argument("--per-page", type=int, default=10)

    code_parser = subparsers.add_parser("code-search", help="Search GitHub code")
    code_parser.add_argument("query")
    code_parser.add_argument("--language", default="")
    code_parser.add_argument("--filename", default="")
    code_parser.add_argument("--extension", default="")
    code_parser.add_argument("--per-page", type=int, default=10)

    info_parser = subparsers.add_parser("repo-info", help="Get repository details")
    info_parser.add_argument("owner")
    info_parser.add_argument("repo")

    tree_parser = subparsers.add_parser("repo-tree", help="Recursively list files in a repository")
    tree_parser.add_argument("owner")
    tree_parser.add_argument("repo")
    tree_parser.add_argument("--ref", default="")
    tree_parser.add_argument("--path-filter", default="")
    tree_parser.add_argument("--max-entries", type=int, default=200)

    fetch_parser = subparsers.add_parser("fetch-file", help="Download a single file from a repository")
    fetch_parser.add_argument("owner")
    fetch_parser.add_argument("repo")
    fetch_parser.add_argument("path")
    fetch_parser.add_argument("--ref", default="")
    fetch_parser.add_argument("--max-bytes", type=int, default=200_000)

    args = parser.parse_args()

    if args.command == "repo-search":
        print(
            search_github_repositories(
                args.query,
                language=args.language,
                sort=args.sort,
                order=args.order,
                per_page=args.per_page,
            )
        )
    elif args.command == "code-search":
        print(
            search_github_code(
                args.query,
                language=args.language,
                filename=args.filename,
                extension=args.extension,
                per_page=args.per_page,
            )
        )
    elif args.command == "repo-info":
        print(get_github_repository_info(args.owner, args.repo))
    elif args.command == "repo-tree":
        import json as _json
        result = list_github_repository_tree(
            args.owner, args.repo,
            ref=args.ref, path_filter=args.path_filter, max_entries=args.max_entries,
        )
        print(_json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "fetch-file":
        import json as _json
        result = fetch_github_file(
            args.owner, args.repo, args.path,
            ref=args.ref, max_bytes=args.max_bytes,
        )
        # Avoid printing huge file contents on the terminal — show metadata + first 800 chars.
        preview = dict(result)
        if isinstance(preview.get("content"), str) and len(preview["content"]) > 800:
            preview["content"] = preview["content"][:800] + "\n...[truncated for CLI display]"
        print(_json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
