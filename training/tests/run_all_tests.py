"""
統合テストスクリプト

Phase 0〜3 + Datasetの全テストを実行して結果をサマリー表示
"""

import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test(test_name: str, test_file: str) -> tuple:
    """
    テストを実行して結果を返す

    引数:
        test_name: テスト名
        test_file: テストファイルパス

    戻り値:
        (success: bool, output: str)
    """
    print(f"\n{'=' * 60}")
    print(f"Running {test_name}...")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=60,
        )

        success = result.returncode == 0
        output = result.stdout if success else result.stderr

        # Print output
        print(output)

        return success, output

    except subprocess.TimeoutExpired:
        print(f"[ERROR] {test_name} timed out!")
        return False, "Timeout"
    except Exception as e:
        print(f"[ERROR] {test_name} failed: {e}")
        return False, str(e)


def main():
    """
    全テスト実行
    """
    print("=" * 60)
    print("SupertonicTTS Training - Integrated Test Suite")
    print("Phase 0-3 + Dataset Full Test Execution")
    print("=" * 60)

    # Test configurations
    tests = [
        ("Phase 0: Speech Autoencoder", "tests/test_autoencoder.py"),
        ("Phase 1: Common Modules", "tests/test_common.py"),
        ("Phase 2: TTL (Text-to-Latent)", "tests/test_ttl.py"),
        ("Phase 3: Duration Predictor", "tests/test_dp.py"),
        ("Dataset: TTSDataset & AudioDataset", "tests/test_dataset.py"),
    ]

    # Run all tests
    results = []
    for test_name, test_file in tests:
        success, output = run_test(test_name, test_file)
        results.append((test_name, success, output))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for _, success, _ in results if success)
    failed_tests = total_tests - passed_tests

    for test_name, success, _ in results:
        status = "[PASSED]" if success else "[FAILED]"
        print(f"{status:10} | {test_name}")

    print("=" * 60)
    print(f"Total: {total_tests} | Passed: {passed_tests} | Failed: {failed_tests}")
    print("=" * 60)

    if failed_tests > 0:
        print("\n[FAILED] Some tests failed. Please review the output above.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
