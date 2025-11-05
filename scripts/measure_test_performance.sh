#!/bin/bash
# テスト実行時間を測定して比較するスクリプト

set -e

PYTHONPATH=$PWD/src:$PYTHONPATH

echo "=========================================="
echo "テストパフォーマンス測定"
echo "=========================================="
echo ""

# テストディレクトリ
TEST_DIR="src/neutryx/tests"

# 測定対象（軽量なテストのみを使用して高速に測定）
SAMPLE_TESTS="${TEST_DIR}/models/test_bs_analytic.py ${TEST_DIR}/products/test_equity.py ${TEST_DIR}/products/test_barrier.py"

echo "1. ベースライン（並列実行なし）"
echo "-------------------------------------------"
time pytest ${SAMPLE_TESTS} -q --tb=no
echo ""

echo "2. 並列実行（-n 2）"
echo "-------------------------------------------"
time pytest ${SAMPLE_TESTS} -n 2 -q --tb=no
echo ""

echo "3. 並列実行（-n auto）"
echo "-------------------------------------------"
time pytest ${SAMPLE_TESTS} -n auto -q --tb=no
echo ""

echo "4. 速いテストのみ（マーカーフィルター）"
echo "-------------------------------------------"
time pytest ${TEST_DIR} -m "fast" -q --tb=no --co -q | tail -1
echo ""

echo "5. 統合テストのみ"
echo "-------------------------------------------"
time pytest ${TEST_DIR}/integration -m "integration" -q --tb=no
echo ""

echo "=========================================="
echo "測定完了"
echo "=========================================="
echo ""
echo "期待される結果："
echo "  - 並列実行により30-50%の時間短縮"
echo "  - マーカーによる選択的実行でフィードバック時間短縮"
echo ""
