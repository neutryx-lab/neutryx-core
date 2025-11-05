# テスト効率化実装ガイド

## 実装内容

### 1. 並列テスト実行（pytest-xdist）
- `pyproject.toml`: pytest-xdist依存関係を追加
- `.github/workflows/ci.yml`: `-n auto`オプションを追加

**使用方法:**
```bash
pytest -n auto        # 全CPUコアを使用
pytest -n 4           # 4コアを使用
```

**期待効果:** 50-70%の時間短縮（4コア以上の環境）

### 2. テストマーカーによる階層化

定義されたマーカー:
- `@pytest.mark.fast`: 0.1秒未満のテスト
- `@pytest.mark.slow`: 1秒以上のテスト
- `@pytest.mark.integration`: 統合テスト
- `@pytest.mark.unit`: ユニットテスト
- `@pytest.mark.regression`: リグレッションテスト
- `@pytest.mark.performance`: パフォーマンステスト

**使用方法:**
```bash
pytest -m "fast"              # 速いテストのみ
pytest -m "not slow"          # 遅いテストを除外
pytest -m "integration"       # 統合テストのみ
pytest -m "unit and fast"     # 速いユニットテスト
```

**実装例:**
```python
@pytest.mark.fast
@pytest.mark.unit
def test_bs_call_atm():
    ...

@pytest.mark.slow
@pytest.mark.unit
def test_calibrate_heston():
    ...

@pytest.mark.integration
class TestPortfolioWorkflow:
    ...
```

### 3. フィクスチャスコープの最適化

```python
# Before: 毎回再作成
@pytest.fixture
def portfolio():
    return create_portfolio()

# After: クラスで1回作成
@pytest.fixture(scope="class")
def portfolio():
    return create_portfolio()
```

### 4. 統合テスト中心のアプローチ

統合テストで複数コンポーネントを同時にテストし、冗長なユニットテストを削減。

**例:** `test_fictional_portfolio.py`が以下をカバー:
- ポートフォリオ作成・管理
- カウンターパーティ設定
- ネッティングセット管理
- XVA計算
- 集計・レポート機能

## 期待される効果

| 最適化手法 | 改善率 | 用途 |
|-----------|--------|------|
| 並列実行（4コア） | 50-70% | CI全体 |
| fastマーカー選択 | 90%+ | 開発中 |
| not slow除外 | 30-50% | PRレビュー |
| フィクスチャ最適化 | 50-80% | セットアップ時間 |

## 次のステップ

1. **全テストにマーカー追加**
   ```bash
   pytest --durations=50  # 実行時間測定
   ```

2. **CI戦略の改善**
   - PR時: 速いテストのみ実行
   - メインブランチ: 全テスト実行

3. **継続的監視**
   - 週次でテスト実行時間をレビュー
   - 新しい遅いテストを特定

## ベストプラクティス

新しいテストを書く際のチェックリスト:
- [ ] 統合テストで十分カバーできないか検討
- [ ] 適切なマーカーを付ける
- [ ] フィクスチャのスコープを最適化
- [ ] テストパラメータは最小限に
- [ ] 並列実行を意識（グローバル状態を避ける）
