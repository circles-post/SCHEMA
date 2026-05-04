"""Step 15: 测试 websearch_tools.py 中 web_search / web_fetch 是否正常工作。

验证点:
- WebSearcher 能否初始化并 start()
- web_search() 返回值字段名是否正确 (urls_fetched / formatted_snippets)
- web_fetch() 私有方法 _fetch_urls 是否可用
- _searcher 单例是否能正常 stop (资源释放)

运行方式:
  python debug_workflow/step15_web_search_tools.py
  python debug_workflow/step15_web_search_tools.py --skip-fetch  # 跳过 web_fetch 测试
"""

import argparse
import asyncio
import inspect
import sys
from pathlib import Path

REPO_DIR = Path("/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")
sys.path.insert(0, str(REPO_DIR))


async def _cleanup_searcher(searcher) -> str | None:
    """Clean up a WebSearcher instance using its supported lifecycle API."""
    for name in ("close", "aclose", "stop"):
        method = getattr(searcher, name, None)
        if method is None or not callable(method):
            continue
        result = method()
        if inspect.isawaitable(result):
            await result
        return name
    return None


async def test_searcher_init():
    """Step 15a: 初始化 WebSearcher 并检查可用的属性/方法。"""
    print("=" * 60)
    print("Step 15a: WebSearcher initialization")
    print("=" * 60)
    try:
        from websearch import WebSearcher
        from websearch_tools import (
            _BRIGHT_DATA_API_KEY,
            _BRIGHT_DATA_ZONE,
            _OPENAI_API_KEY,
            _OPENAI_BASE_URL,
        )

        searcher = WebSearcher(
            bright_data_api_key=_BRIGHT_DATA_API_KEY or None,
            bright_data_zone=_BRIGHT_DATA_ZONE or None,
            openai_api_key=_OPENAI_API_KEY or None,
            openai_base_url=_OPENAI_BASE_URL or None,
        )
        print(f"  WebSearcher instance: {searcher}")
        await searcher.start()
        print("  [OK] searcher.start() succeeded")

        # 检查 search 方法签名
        search_sig = inspect.signature(searcher.search)
        print(f"  searcher.search signature: {search_sig}")

        # 检查是否有 _fetch_urls 私有方法 (web_fetch 依赖它)
        has_private = hasattr(searcher, "_fetch_urls") and callable(searcher._fetch_urls)
        if has_private:
            fetch_sig = inspect.signature(searcher._fetch_urls)
            print(f"  searcher._fetch_urls signature: {fetch_sig}")
            print("  [OK] _fetch_urls private method exists")
        else:
            print("  [WARN] _fetch_urls NOT found. web_fetch() will fail!")
            # 列出所有方法供排查
            methods = [m for m in dir(searcher) if "fetch" in m.lower() or "url" in m.lower()]
            print(f"  Available fetch/url methods: {methods}")

        # 检查清理方法
        has_stop = hasattr(searcher, "stop") and callable(searcher.stop)
        has_close = hasattr(searcher, "close") and callable(searcher.close)
        print(f"  searcher.stop exists: {has_stop}")
        print(f"  searcher.close exists: {has_close}")
        cleanup_method = await _cleanup_searcher(searcher)
        if cleanup_method is not None:
            print(f"  [OK] searcher.{cleanup_method}() succeeded")
        else:
            print("  [WARN] WebSearcher has no usable cleanup method - resource leak risk")

        return searcher, has_private, has_stop
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return None, False, False


async def test_web_search():
    """Step 15b: 调用 web_search() 并检查返回值结构。"""
    print("\n" + "=" * 60)
    print("Step 15b: web_search() function")
    print("=" * 60)

    # 先直接检查 WebSearcher.search 返回对象的字段
    from websearch import WebSearcher
    from websearch_tools import (
        _BRIGHT_DATA_API_KEY,
        _BRIGHT_DATA_ZONE,
        _OPENAI_API_KEY,
        _OPENAI_BASE_URL,
    )

    searcher = WebSearcher(
        bright_data_api_key=_BRIGHT_DATA_API_KEY or None,
        bright_data_zone=_BRIGHT_DATA_ZONE or None,
        openai_api_key=_OPENAI_API_KEY or None,
        openai_base_url=_OPENAI_BASE_URL or None,
    )
    await searcher.start()

    try:
        print("  Calling searcher.search('protein structure PDB', num_results=3)...")
        result = await searcher.search("protein structure PDB", num_results=3)

        print(f"  Result type: {type(result)}")
        print(f"  Result attributes: {[a for a in dir(result) if not a.startswith('__')]}")

        # 检查 web_search 使用的字段
        has_urls_fetched = hasattr(result, "urls_fetched")
        has_formatted_snippets = hasattr(result, "formatted_snippets")
        print(f"  result.urls_fetched exists: {has_urls_fetched}")
        print(f"  result.formatted_snippets exists: {has_formatted_snippets}")

        if has_urls_fetched:
            print(f"  urls_fetched: {result.urls_fetched}")
        else:
            print("  [FAIL] urls_fetched field missing - web_search() will crash at L71!")
            all_attrs = {a: getattr(result, a) for a in dir(result) if not a.startswith('__')}
            print(f"  All fields: {list(all_attrs.keys())}")

        if has_formatted_snippets:
            snippet_preview = str(result.formatted_snippets)[:300]
            print(f"  formatted_snippets (first 300 chars): {snippet_preview}")
        else:
            print("  [FAIL] formatted_snippets field missing - web_search() will crash at L80!")

    except Exception as e:
        print(f"  [FAIL] searcher.search() failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await _cleanup_searcher(searcher)


async def test_web_search_tool():
    """Step 15c: 通过 websearch_tools.web_search() 高级接口测试。"""
    print("\n" + "=" * 60)
    print("Step 15c: websearch_tools.web_search() high-level function")
    print("=" * 60)
    try:
        from websearch_tools import web_search
        result_str = await web_search("hemoglobin oxygen transport", num_results=3)
        print(f"  Return type: {type(result_str)}")
        print(f"  Output (first 600 chars):\n{result_str[:600]}")
        assert isinstance(result_str, str), "Expected str return type"
        assert len(result_str) > 10, "Result seems too short"
        print("  [OK] web_search() returned valid string")
    except Exception as e:
        print(f"  [FAIL] web_search() failed: {e}")
        import traceback
        traceback.print_exc()


async def test_web_fetch_tool(skip: bool = False):
    """Step 15d: 测试 web_fetch() 私有方法路径。"""
    print("\n" + "=" * 60)
    print("Step 15d: websearch_tools.web_fetch() high-level function")
    print("=" * 60)
    if skip:
        print("  [SKIP] --skip-fetch specified")
        return

    try:
        from websearch_tools import web_fetch
        # 使用一个稳定的 URL
        test_url = "https://en.wikipedia.org/wiki/Hemoglobin"
        print(f"  Fetching: {test_url}")
        result_str = await web_fetch([test_url])
        print(f"  Return type: {type(result_str)}")
        print(f"  Output (first 500 chars):\n{result_str[:500]}")
        assert isinstance(result_str, str)
        assert "Could not fetch" not in result_str or len(result_str) > 100
        print("  [OK] web_fetch() returned content")
    except AttributeError as e:
        print(f"  [FAIL] AttributeError (likely _fetch_urls is missing): {e}")
    except Exception as e:
        print(f"  [FAIL] web_fetch() failed: {e}")
        import traceback
        traceback.print_exc()


async def test_stop_singleton():
    """Step 15e: 验证单例是否能被 stop 释放。"""
    print("\n" + "=" * 60)
    print("Step 15e: _searcher singleton cleanup")
    print("=" * 60)
    import websearch_tools
    # 触发单例初始化
    await websearch_tools._get_searcher()
    searcher = websearch_tools._searcher
    print(f"  Singleton created: {searcher is not None}")

    if searcher is not None:
        cleanup_method = await _cleanup_searcher(searcher)
        if cleanup_method is not None:
            websearch_tools._searcher = None
            print(f"  [OK] Singleton cleaned up via {cleanup_method}() and reset")
        else:
            print("  [FAIL] No cleanup method found - resource will leak")


async def main(skip_fetch: bool = False):
    await test_searcher_init()
    await test_web_search()
    await test_web_search_tool()
    await test_web_fetch_tool(skip=skip_fetch)
    await test_stop_singleton()

    print("\n" + "=" * 60)
    print("[DONE] Step 15 completed. Review any [FAIL]/[WARN] above.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip web_fetch test (avoids external HTTP calls)")
    args = parser.parse_args()
    asyncio.run(main(skip_fetch=args.skip_fetch))
