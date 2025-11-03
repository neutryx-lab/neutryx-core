# Neutryx Core - Public Release Checklist

このドキュメントは、Neutryx CoreをPublicリリースする前に完了すべき項目のチェックリストです。

## ✅ 完了済み

### パッケージとドキュメントの整備
- [x] パッケージ名を `neutryx-core` に統一
- [x] メールアドレスを `dev@neutryx.tech` に修正
- [x] 著作権年を2025に統一
- [x] GitHubリポジトリURLを `neutryx-lab/neutryx-core` に修正
- [x] CHANGELOG.md作成
- [x] CONTRIBUTING.md作成
- [x] CODE_OF_CONDUCT.md作成
- [x] docs/security_audit.md作成
- [x] docs/roadmap.md作成

### CI/CDとGitHub設定
- [x] CI/CDワークフロー確認（.github/workflows/ci.yml）
- [x] Dependabot設定確認（.github/dependabot.yml）
- [x] ドキュメント公開ワークフロー確認（.github/workflows/docs.yml）
- [x] Issueテンプレート作成（bug, feature, question）
- [x] Pull Requestテンプレート作成

### セキュリティとテスト
- [x] 依存関係の脆弱性チェック実施（pip-audit - 問題なし）
- [x] 基本的なテストの実行確認

## 📋 GitHub上で実施する必要がある設定

リポジトリをPublicにする**前**に、以下の設定をGitHub上で行ってください：

### 1. リポジトリ設定

#### General Settings
1. リポジトリの説明を設定:
   ```
   A compact JAX-based quant finance library (pricing, risk, XVA, calibration)
   ```

2. トピックを追加:
   - `quantitative-finance`
   - `jax`
   - `pricing`
   - `risk-management`
   - `python`
   - `derivatives`
   - `monte-carlo`
   - `machine-learning`

3. ウェブサイトURLを設定:
   ```
   https://neutryx.tech
   ```

#### Features
- [x] **Wikis**: 無効化（ドキュメントはdocs/で管理）
- [x] **Issues**: 有効化
- [x] **Sponsorship**: 必要に応じて設定
- [x] **Projects**: 必要に応じて有効化
- [x] **Discussions**: 有効化（コミュニティサポート用）

### 2. ブランチ保護ルール

`Settings` → `Branches` → `Add rule` で `main` ブランチに以下のルールを設定：

- [x] **Require a pull request before merging**
  - Require approvals: 1
  - Dismiss stale pull request approvals when new commits are pushed
  
- [x] **Require status checks to pass before merging**
  - Require branches to be up to date before merging
  - Status checks required:
    - `unit-tests`
    - `security`
    
- [x] **Require conversation resolution before merging**

- [x] **Do not allow bypassing the above settings**

### 3. セキュリティ設定

`Settings` → `Security` で以下を設定：

#### Dependabot
- [x] **Dependabot alerts**: 有効化
- [x] **Dependabot security updates**: 有効化

#### Code scanning
- [x] **CodeQL analysis**: 設定（推奨）
  ```bash
  # GitHub UI から "Set up CodeQL" をクリック
  # デフォルト設定で有効化
  ```

#### Secret scanning
- [x] **Secret scanning**: 有効化（GitHub ProまたはPublicリポジトリで利用可能）

### 4. GitHub Pages設定

`Settings` → `Pages` で以下を設定：

- **Source**: GitHub Actions
- **Custom domain** (オプション): 必要に応じて設定

ドキュメントは以下のURLで公開されます：
```
https://neutryx-lab.github.io/neutryx-core/
```

### 5. Discussions設定

`Settings` → `General` → `Features` → `Discussions` を有効化

推奨カテゴリ:
- **Q&A**: 質問と回答
- **Ideas**: 機能提案とアイデア
- **Show and tell**: ユーザーの実装例
- **General**: 一般的な議論
- **Announcements**: 重要なお知らせ

### 6. Labels設定

`Issues` → `Labels` で以下のラベルを追加：

**Type:**
- `bug` (red) - バグ報告
- `enhancement` (light blue) - 機能追加
- `documentation` (blue) - ドキュメント関連
- `question` (purple) - 質問

**Priority:**
- `priority: critical` (dark red)
- `priority: high` (orange)
- `priority: medium` (yellow)
- `priority: low` (light gray)

**Area:**
- `area: models` - モデル関連
- `area: pricing` - 価格計算
- `area: risk` - リスク管理
- `area: xva` - XVA計算
- `area: calibration` - キャリブレーション
- `area: infrastructure` - インフラ・CI/CD

**Status:**
- `good first issue` (green) - 初心者向け
- `help wanted` (green) - コントリビューション歓迎
- `wontfix` (gray) - 対応しない
- `duplicate` (gray) - 重複

## 🚀 リリース手順

### 1. 最終確認
```bash
# 全変更をステージング
git add .

# 変更内容の確認
git status
git diff --cached

# コミット
git commit -m "chore: prepare for public release

- Update package name to neutryx-core
- Fix email addresses and repository URLs
- Add comprehensive documentation (CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT)
- Add GitHub templates (issues, PR)
- Update security and roadmap documentation
"

# プッシュ
git push origin main
```

### 2. リポジトリをPublicに変更

`Settings` → `General` → `Danger Zone` → `Change visibility` → `Make public`

**注意**: この操作は取り消せません。必ず上記のチェックリストをすべて完了してから実行してください。

### 3. GitHubリリースの作成

1. `Releases` → `Create a new release`をクリック
2. タグ: `v0.1.0`
3. リリースタイトル: `v0.1.0 - Initial Public Release`
4. 説明:
   ```markdown
   # Neutryx Core v0.1.0 - Initial Public Release

   🎉 Welcome to the first public release of Neutryx Core!

   ## Overview

   Neutryx Core is a JAX-based quantitative finance library designed for
   high-performance pricing, risk management, and calibration.

   ## Features

   - ✅ Black-Scholes analytical pricing and Greeks
   - ✅ Monte Carlo simulation with JAX JIT compilation
   - ✅ Multiple stochastic models (Heston, SABR, Jump Diffusion)
   - ✅ Path-dependent options (Asian, Barrier, Lookback, American)
   - ✅ XVA suite (CVA, FVA, MVA)
   - ✅ GPU/TPU optimization
   - ✅ Differentiable calibration framework

   ## Installation

   ```bash
   pip install neutryx-core
   ```

   ## Documentation

   - [README](https://github.com/neutryx-lab/neutryx-core/blob/main/README.md)
   - [Contributing Guide](https://github.com/neutryx-lab/neutryx-core/blob/main/CONTRIBUTING.md)
   - [API Documentation](https://neutryx-lab.github.io/neutryx-core/)

   ## What's Next

   See our [roadmap](https://github.com/neutryx-lab/neutryx-core/blob/main/docs/roadmap.md)
   for planned features and upcoming releases.

   ## Contributors

   Built with ❤️ by the Neutryx Lab team
   ```

5. `Publish release`をクリック

### 4. PyPIへの公開（オプション）

PyPIに公開する場合：

```bash
# パッケージのビルド
python -m build

# TestPyPIでテスト
twine upload --repository testpypi dist/*

# 本番PyPIへアップロード
twine upload dist/*
```

### 5. 告知

リリース後、以下の場所で告知：

- [ ] GitHub Discussions に投稿
- [ ] Twitter/X でアナウンス（@neutryx_lab）
- [ ] LinkedIn で共有
- [ ] Reddit (r/quant, r/Python) に投稿
- [ ] ブログ記事の公開

## 📊 モニタリング

リリース後、以下を定期的に確認：

- [ ] GitHub Issues の対応
- [ ] GitHub Discussions の回答
- [ ] CI/CDの動作確認
- [ ] Dependabot PR の確認とマージ
- [ ] セキュリティアラートの確認
- [ ] ドキュメントの更新

## 📞 サポート

問題が発生した場合：
- Email: dev@neutryx.tech
- GitHub Issues: https://github.com/neutryx-lab/neutryx-core/issues
- GitHub Discussions: https://github.com/neutryx-lab/neutryx-core/discussions

---

**Last Updated**: 2025-01-03
**Version**: 0.1.0
