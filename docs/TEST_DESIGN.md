# テスト設計ドキュメント - CareerNavigator

**バージョン**: 1.0
**作成日**: 2025-10-23
**対象プロジェクト**: skillnote-recommendation（スキルノート推薦システム）

---

## 目次

1. [概要](#概要)
2. [テスト戦略](#テスト戦略)
3. [テスト構成](#テスト構成)
4. [コンポーネント別テスト設計](#コンポーネント別テスト設計)
5. [テストデータ戦略](#テストデータ戦略)
6. [テスト実行環境](#テスト実行環境)
7. [カバレッジ目標](#カバレッジ目標)
8. [実装優先度](#実装優先度)

---

## 1. 概要

### 1.1 目的

このドキュメントは、CareerNavigator（スキルノート推薦システム）の包括的なテスト設計を定義します。品質保証とリグレッション防止のため、各コンポーネントに対する詳細なテストケースを提供します。

### 1.2 対象システム

**システム名**: スキルノート推薦システム
**主要機能**:
- 技術者のスキルプロファイル分析
- 力量（コンピテンシー）の推薦
- 同時習得パターンの検出
- 推薦理由の生成

**主要コンポーネント**:
1. データモデル（`models.py`）
2. データローダー（`data_loader.py`）
3. データ変換（`data_transformer.py`）
4. 類似度計算（`similarity_calculator.py`）
5. 推薦エンジン（`recommendation_engine.py`）
6. 推薦システム（`recommendation_system.py`）

### 1.3 現在のテスト状況

**既存テスト**: `tests/test_basic.py`（8テストケース）
**カバレッジ**: 最小限（モデルとConfigのみ）
**課題**:
- データパイプラインのテストが不足
- 推薦アルゴリズムのテストがない
- 統合テストがない
- エッジケースのテストがない

---

## 2. テスト戦略

### 2.1 テストレベル

| レベル | 目的 | スコープ |
|--------|------|----------|
| **単体テスト** | 個別関数/メソッドの正確性検証 | 全コンポーネント（80%以上のカバレッジ目標）|
| **統合テスト** | コンポーネント間の連携検証 | データパイプライン全体 |
| **E2Eテスト** | エンドツーエンドシナリオ検証 | データ読込→変換→推薦→出力 |

### 2.2 テストアプローチ

- **TDD（テスト駆動開発）**: 新規機能実装時に推奨
- **リグレッションテスト**: CI/CD パイプラインで自動実行
- **境界値分析**: 数値計算、レベル正規化等で実施
- **等価分割**: 力量タイプ（SKILL/EDUCATION/LICENSE）ごとに検証

### 2.3 テストツール

- **テストフレームワーク**: pytest 7.0+
- **モック/スタブ**: pytest-mock, unittest.mock
- **フィクスチャ**: pytest fixtures（conftest.py）
- **カバレッジ**: pytest-cov
- **データ生成**: pandas, numpy（テストデータ作成用）

---

## 3. テスト構成

### 3.1 テストディレクトリ構造

```
tests/
├── __init__.py
├── conftest.py                      # 共通フィクスチャ
├── test_models.py                   # データモデルテスト
├── test_data_loader.py              # データローダーテスト
├── test_data_transformer.py         # データ変換テスト
├── test_similarity_calculator.py    # 類似度計算テスト
├── test_recommendation_engine.py    # 推薦エンジンテスト
├── test_recommendation_system.py    # 推薦システムテスト
├── test_integration.py              # 統合テスト
├── test_e2e.py                      # E2Eテスト
├── fixtures/                        # テストデータフィクスチャ
│   ├── sample_members.csv
│   ├── sample_skills.csv
│   ├── sample_acquired.csv
│   └── ...
└── helpers/                         # テストヘルパー関数
    ├── __init__.py
    └── data_factory.py              # テストデータ生成ヘルパー
```

---

## 4. コンポーネント別テスト設計

### 4.1 データモデル（`models.py`）

#### テストファイル: `test_models.py`

#### テストケース

| # | テストケース | 検証内容 | 優先度 |
|---|------------|---------|--------|
| 1 | `test_member_creation` | Member インスタンス作成とプロパティ確認 | 高 |
| 2 | `test_member_repr` | Member の文字列表現が正しいこと | 中 |
| 3 | `test_member_with_optional_fields` | role, grade が None でも動作すること | 高 |
| 4 | `test_competence_creation` | Competence インスタンス作成とプロパティ確認 | 高 |
| 5 | `test_competence_types` | 3種類の力量タイプ（SKILL/EDUCATION/LICENSE）が設定可能 | 高 |
| 6 | `test_competence_with_optional_fields` | category, description が None でも動作 | 中 |
| 7 | `test_member_competence_creation` | MemberCompetence インスタンス作成 | 高 |
| 8 | `test_member_competence_level_range` | level が 0-5 の範囲であること | 高 |
| 9 | `test_recommendation_creation` | Recommendation インスタンス作成 | 高 |
| 10 | `test_recommendation_to_dict` | to_dict() メソッドの変換結果検証 | 高 |
| 11 | `test_recommendation_dict_keys` | 辞書に必須キーがすべて含まれること | 高 |
| 12 | `test_recommendation_score_rounding` | スコアが小数第2位まで丸められること | 中 |

#### サンプルテストコード

```python
def test_member_with_optional_fields():
    """オプションフィールドがNoneでもMemberが作成できること"""
    member = Member(member_code='m001', name='テスト太郎')

    assert member.member_code == 'm001'
    assert member.name == 'テスト太郎'
    assert member.role is None
    assert member.grade is None

def test_competence_types():
    """3種類の力量タイプが正しく設定できること"""
    types = ['SKILL', 'EDUCATION', 'LICENSE']

    for comp_type in types:
        comp = Competence(
            competence_code=f'c_{comp_type}',
            name=f'Test {comp_type}',
            competence_type=comp_type
        )
        assert comp.competence_type == comp_type
```

---

### 4.2 データローダー（`data_loader.py`）

#### テストファイル: `test_data_loader.py`

#### テストケース

| # | テストケース | 検証内容 | 優先度 |
|---|------------|---------|--------|
| 1 | `test_clean_column_name` | カラム名のクリーニング（###[xxx]### 除去） | 高 |
| 2 | `test_clean_column_name_with_spaces` | 前後の空白除去 | 高 |
| 3 | `test_clean_column_name_multiple_markers` | 複数のマーカーが含まれる場合 | 中 |
| 4 | `test_load_csv_success` | 正常なCSVファイル読み込み | 高 |
| 5 | `test_load_csv_file_not_found` | 存在しないファイルで FileNotFoundError | 高 |
| 6 | `test_load_csv_encoding` | 文字エンコーディング（UTF-8-sig）の処理 | 高 |
| 7 | `test_load_csv_column_cleaning` | 読み込み時にカラム名が自動クリーニングされる | 高 |
| 8 | `test_load_all_data_success` | 全ファイルの一括読み込み成功 | 高 |
| 9 | `test_load_all_data_missing_file` | 1つでもファイルが欠けていたら例外 | 高 |
| 10 | `test_validate_data_success` | 正常データの検証が True を返す | 高 |
| 11 | `test_validate_data_missing_columns` | 必須カラム欠落時に False を返す | 高 |
| 12 | `test_validate_data_each_file` | 各ファイルタイプの必須カラム確認 | 高 |
| 13 | `test_loader_custom_data_dir` | カスタムデータディレクトリの指定 | 中 |

#### サンプルテストコード

```python
def test_clean_column_name():
    """カラム名のクリーニング機能"""
    # マーカー付きカラム名
    assert DataLoader.clean_column_name('メンバー名 ###[Member Name]###') == 'メンバー名'

    # 前後に空白
    assert DataLoader.clean_column_name('  力量コード  ') == '力量コード'

    # 両方
    assert DataLoader.clean_column_name('  レベル ###[Level]###  ') == 'レベル'

def test_load_csv_file_not_found(tmp_path):
    """存在しないファイルを読み込むとFileNotFoundError"""
    loader = DataLoader(data_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError):
        loader.load_csv('non_existent.csv')
```

---

### 4.3 データ変換（`data_transformer.py`）

#### テストファイル: `test_data_transformer.py`

#### テストケース

| # | テストケース | 検証内容 | 優先度 |
|---|------------|---------|--------|
| 1 | `test_normalize_level_skill_valid` | SKILLタイプの正規化（1-5の整数） | 高 |
| 2 | `test_normalize_level_skill_invalid` | SKILL で変換失敗時は 0 | 高 |
| 3 | `test_normalize_level_education_marked` | EDUCATION で ●→1 に変換 | 高 |
| 4 | `test_normalize_level_education_empty` | EDUCATION で空文字→0 | 高 |
| 5 | `test_normalize_level_license_marked` | LICENSE で ●→1 に変換 | 高 |
| 6 | `test_normalize_level_license_nan` | LICENSE で NaN→0 | 高 |
| 7 | `test_create_competence_master` | 統合力量マスタの作成 | 高 |
| 8 | `test_competence_master_types` | 3タイプがすべて含まれること | 高 |
| 9 | `test_competence_master_level_ranges` | レベル範囲（'1-5'または'●'）が正しい | 中 |
| 10 | `test_create_category_names` | カテゴリ名マッピングの作成 | 高 |
| 11 | `test_create_category_names_hierarchy` | 階層カテゴリの結合（' > '区切り） | 高 |
| 12 | `test_create_member_competence` | メンバー習得力量データの作成 | 高 |
| 13 | `test_filter_invalid_members` | 削除・テストユーザーの除外 | 高 |
| 14 | `test_member_competence_merge` | 力量マスタとのマージ | 高 |
| 15 | `test_create_skill_matrix` | メンバー×力量マトリクスの作成 | 高 |
| 16 | `test_skill_matrix_shape` | マトリクスの行列数確認 | 高 |
| 17 | `test_skill_matrix_fill_value` | 未習得箇所が 0 で埋められる | 高 |
| 18 | `test_clean_members_data` | メンバーマスタのクリーニング | 中 |

#### サンプルテストコード

```python
@pytest.mark.parametrize("level,expected", [
    ('1', 1), ('3', 3), ('5', 5),
    ('invalid', 0), ('', 0), (None, 0)
])
def test_normalize_level_skill(level, expected):
    """SKILLタイプのレベル正規化"""
    result = DataTransformer.normalize_level(level, 'SKILL')
    assert result == expected

@pytest.mark.parametrize("comp_type,level,expected", [
    ('EDUCATION', '●', 1),
    ('EDUCATION', '', 0),
    ('EDUCATION', None, 0),
    ('LICENSE', '●', 1),
    ('LICENSE', '', 0),
])
def test_normalize_level_non_skill(comp_type, level, expected):
    """EDUCATION/LICENSEタイプのレベル正規化"""
    result = DataTransformer.normalize_level(level, comp_type)
    assert result == expected
```

---

### 4.4 類似度計算（`similarity_calculator.py`）

#### テストファイル: `test_similarity_calculator.py`

#### テストケース

| # | テストケース | 検証内容 | 優先度 |
|---|------------|---------|--------|
| 1 | `test_calculator_initialization` | デフォルトパラメータでの初期化 | 中 |
| 2 | `test_calculator_custom_params` | カスタムパラメータでの初期化 | 中 |
| 3 | `test_calculate_similarity_basic` | 基本的な類似度計算 | 高 |
| 4 | `test_jaccard_coefficient` | Jaccard係数の正確性（手計算と比較） | 高 |
| 5 | `test_similarity_threshold` | 閾値以下の類似度は除外される | 高 |
| 6 | `test_similarity_symmetric` | (A,B) のペアのみで (B,A) は含まれない | 高 |
| 7 | `test_similarity_no_acquirers` | 習得者ゼロの力量はスキップ | 高 |
| 8 | `test_similarity_sampling` | サンプリングサイズが適用される | 中 |
| 9 | `test_similarity_identical_skills` | 完全に同じ習得パターン（類似度=1.0） | 中 |
| 10 | `test_similarity_no_overlap` | 全く重複なし（類似度=0） | 中 |
| 11 | `test_similarity_output_format` | 出力DataFrameの形式確認 | 高 |
| 12 | `test_empty_data` | 空データでも例外が発生しない | 中 |

#### サンプルテストコード

```python
def test_jaccard_coefficient():
    """Jaccard係数の正確性を手計算で検証"""
    # テストデータ作成
    data = pd.DataFrame({
        'メンバーコード': ['m1', 'm1', 'm2', 'm2', 'm3'],
        '力量コード': ['s1', 's2', 's1', 's3', 's2'],
        '正規化レベル': [3, 4, 2, 5, 3]
    })

    # s1: {m1, m2}
    # s2: {m1, m3}
    # s3: {m2}
    # Jaccard(s1, s2) = |{m1}| / |{m1,m2,m3}| = 1/3 = 0.333...

    calculator = SimilarityCalculator(sample_size=100, threshold=0.1)
    result = calculator.calculate_similarity(data)

    s1_s2_similarity = result[
        ((result['力量1'] == 's1') & (result['力量2'] == 's2')) |
        ((result['力量1'] == 's2') & (result['力量2'] == 's1'))
    ]['類似度'].values[0]

    assert abs(s1_s2_similarity - 1/3) < 0.01
```

---

### 4.5 推薦エンジン（`recommendation_engine.py`）

#### テストファイル: `test_recommendation_engine.py`

#### テストケース

| # | テストケース | 検証内容 | 優先度 |
|---|------------|---------|--------|
| 1 | `test_engine_initialization` | 必要なDataFrameでの初期化 | 高 |
| 2 | `test_get_member_competences` | メンバーの保有力量取得 | 高 |
| 3 | `test_get_member_competences_empty` | 力量未保有メンバーで空データ返却 | 中 |
| 4 | `test_get_unacquired_competences` | 未習得力量の取得 | 高 |
| 5 | `test_unacquired_with_type_filter` | 力量タイプフィルタの適用 | 高 |
| 6 | `test_calculate_category_importance` | カテゴリ重要度の計算（0-10範囲） | 高 |
| 7 | `test_category_importance_edge_cases` | カテゴリ内習得者ゼロ時のデフォルト値 | 中 |
| 8 | `test_calculate_acquisition_ease` | 習得容易性の計算 | 高 |
| 9 | `test_acquisition_ease_with_similar` | 類似力量保有時に高スコア | 高 |
| 10 | `test_acquisition_ease_no_similar` | 類似力量なし時のデフォルト値（3.0） | 中 |
| 11 | `test_calculate_popularity` | 人気度の計算（0-10範囲） | 高 |
| 12 | `test_popularity_zero_acquirers` | 習得者ゼロで 0.0 を返す | 中 |
| 13 | `test_calculate_priority_score` | 優先度スコアの計算式 | 高 |
| 14 | `test_priority_score_weights` | 重みの合計が1.0であること | 高 |
| 15 | `test_generate_recommendation_reason` | 推薦理由の生成 | 高 |
| 16 | `test_reason_high_category_importance` | 高カテゴリ重要度時の理由文言 | 中 |
| 17 | `test_reason_high_acquisition_ease` | 高習得容易性時の理由文言 | 中 |
| 18 | `test_reason_by_competence_type` | 力量タイプ別の理由文言 | 中 |
| 19 | `test_recommend_basic` | 基本的な推薦実行 | 高 |
| 20 | `test_recommend_top_n` | top_n パラメータの動作 | 高 |
| 21 | `test_recommend_sorted_by_priority` | 優先度降順でソート | 高 |
| 22 | `test_recommend_with_type_filter` | 力量タイプフィルタ適用 | 高 |
| 23 | `test_recommend_with_category_filter` | カテゴリフィルタ適用 | 高 |
| 24 | `test_recommend_no_results` | 推薦可能力量なし時に空リスト | 中 |

#### サンプルテストコード

```python
def test_calculate_priority_score(sample_engine):
    """優先度スコア計算式の検証"""
    # デフォルト重み: category=0.4, ease=0.3, popularity=0.3
    cat_importance = 8.0
    ease = 6.0
    popularity = 5.0

    expected = 8.0*0.4 + 6.0*0.3 + 5.0*0.3
    result = sample_engine.calculate_priority_score(cat_importance, ease, popularity)

    assert abs(result - expected) < 0.01

def test_recommend_sorted_by_priority(sample_engine):
    """推薦結果が優先度降順でソートされること"""
    recommendations = sample_engine.recommend('m001', top_n=10)

    scores = [rec.priority_score for rec in recommendations]
    assert scores == sorted(scores, reverse=True)
```

---

### 4.6 推薦システム（`recommendation_system.py`）

#### テストファイル: `test_recommendation_system.py`

#### テストケース

| # | テストケース | 検証内容 | 優先度 |
|---|------------|---------|--------|
| 1 | `test_system_initialization` | システム初期化とデータ読み込み | 高 |
| 2 | `test_initialization_missing_files` | データファイル欠落時に例外 | 高 |
| 3 | `test_get_member_info` | メンバー情報取得 | 高 |
| 4 | `test_get_member_info_not_found` | 存在しないメンバーでNone返却 | 中 |
| 5 | `test_member_info_structure` | 返却辞書の構造確認 | 高 |
| 6 | `test_member_info_competence_counts` | 保有力量数の集計 | 高 |
| 7 | `test_recommend_competences` | 力量推薦の実行 | 高 |
| 8 | `test_recommend_with_filters` | フィルタ適用後の推薦 | 高 |
| 9 | `test_print_recommendations` | 推薦結果の表示（出力検証） | 中 |
| 10 | `test_export_recommendations` | CSV出力機能 | 高 |
| 11 | `test_export_file_content` | 出力CSVの内容検証 | 高 |
| 12 | `test_export_encoding` | 出力エンコーディング確認 | 中 |

#### サンプルテストコード

```python
def test_get_member_info(sample_system):
    """メンバー情報取得の検証"""
    info = sample_system.get_member_info('m001')

    assert info is not None
    assert 'member_code' in info
    assert 'name' in info
    assert 'skill_count' in info
    assert 'education_count' in info
    assert 'license_count' in info
    assert info['member_code'] == 'm001'
    assert isinstance(info['skill_count'], int)

def test_export_recommendations(sample_system, tmp_path):
    """CSV出力機能の検証"""
    output_file = 'test_recommendations.csv'
    sample_system.output_dir = str(tmp_path)

    sample_system.export_recommendations('m001', output_file, top_n=5)

    output_path = tmp_path / output_file
    assert output_path.exists()

    df = pd.read_csv(output_path, encoding='utf-8-sig')
    assert len(df) <= 5
    assert '力量名' in df.columns
    assert '優先度スコア' in df.columns
```

---

### 4.7 統合テスト（`test_integration.py`）

#### テストケース

| # | テストケース | 検証内容 | 優先度 |
|---|------------|---------|--------|
| 1 | `test_data_pipeline_end_to_end` | データ読込→変換→類似度計算の一連の流れ | 高 |
| 2 | `test_recommendation_pipeline` | データ準備→推薦実行の流れ | 高 |
| 3 | `test_invalid_member_handling` | 無効なメンバーコードのエラーハンドリング | 高 |
| 4 | `test_empty_data_handling` | 空データでのシステム動作 | 中 |
| 5 | `test_large_dataset_performance` | 大規模データでのパフォーマンス | 低 |

---

### 4.8 E2Eテスト（`test_e2e.py`）

#### テストケース

| # | テストケース | 検証内容 | 優先度 |
|---|------------|---------|--------|
| 1 | `test_convert_data_script` | convert_data スクリプトの実行 | 高 |
| 2 | `test_recommend_script` | run_recommendation スクリプトの実行 | 高 |
| 3 | `test_full_workflow` | データ変換→推薦→CSV出力の完全フロー | 高 |
| 4 | `test_output_files_created` | 全出力ファイルが正しく生成される | 高 |
| 5 | `test_cli_with_arguments` | コマンドライン引数での実行 | 中 |

---

## 5. テストデータ戦略

### 5.1 テストデータの種類

| 種類 | 目的 | データ量 |
|------|------|----------|
| **最小データセット** | 基本機能の検証 | メンバー5名、力量20件 |
| **正常データセット** | 実際の運用に近いデータ | メンバー50名、力量100件 |
| **エッジケースデータ** | 境界値・例外ケース | 各種異常パターン |
| **パフォーマンステストデータ** | スケーラビリティ検証 | メンバー500名、力量500件 |

### 5.2 テストフィクスチャ

#### conftest.py での共通フィクスチャ定義

```python
@pytest.fixture
def sample_members():
    """サンプルメンバーデータ"""
    return pd.DataFrame({
        'メンバーコード': ['m001', 'm002', 'm003'],
        'メンバー名': ['田中太郎', '鈴木花子', '佐藤次郎'],
        '役職': ['主任', '係長', 'スタッフ'],
        '職能・等級': ['3等級', '4等級', '2等級']
    })

@pytest.fixture
def sample_competences():
    """サンプル力量マスタ"""
    return pd.DataFrame({
        '力量コード': ['s001', 's002', 'e001', 'l001'],
        '力量名': ['Python', 'SQL', 'AWS研修', '情報処理技術者'],
        '力量タイプ': ['SKILL', 'SKILL', 'EDUCATION', 'LICENSE'],
        '力量カテゴリー名': ['プログラミング', 'データベース', 'クラウド', '資格']
    })

@pytest.fixture
def sample_member_competence():
    """サンプル習得データ"""
    return pd.DataFrame({
        'メンバーコード': ['m001', 'm001', 'm002', 'm003'],
        '力量コード': ['s001', 's002', 's001', 's002'],
        '正規化レベル': [3, 4, 2, 5],
        '力量タイプ': ['SKILL', 'SKILL', 'SKILL', 'SKILL'],
        '力量カテゴリー名': ['プログラミング', 'データベース', 'プログラミング', 'データベース']
    })
```

### 5.3 エッジケースパターン

1. **空データ**: メンバーゼロ、力量ゼロ
2. **単一データ**: メンバー1名のみ、力量1件のみ
3. **重複データ**: 同一コードの重複
4. **欠損値**: NaN, 空文字列
5. **境界値**: レベル0, レベル5, 類似度0.0, 類似度1.0
6. **文字化け**: 不正なエンコーディング
7. **特殊文字**: カンマ、改行、引用符を含むデータ

---

## 6. テスト実行環境

### 6.1 環境構築

```bash
# 開発依存関係のインストール
uv sync --dev

# または pip の場合
pip install -e ".[dev]"
```

### 6.2 テスト実行コマンド

```bash
# 全テスト実行
uv run pytest tests/

# カバレッジ付き実行
uv run pytest --cov=skillnote_recommendation --cov-report=html tests/

# 特定のファイルのみ
uv run pytest tests/test_models.py

# マーカーで絞り込み
uv run pytest -m "not slow"  # slowマーク以外を実行

# 詳細モード
uv run pytest -v tests/
```

### 6.3 CI/CD パイプライン

GitHub Actions 等での自動テスト実行を推奨：

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install uv
      - run: uv sync --dev
      - run: uv run pytest --cov=skillnote_recommendation tests/
```

---

## 7. カバレッジ目標

### 7.1 全体目標

| メトリクス | 目標値 | 現状 |
|-----------|--------|------|
| **行カバレッジ** | 80%以上 | 未測定 |
| **分岐カバレッジ** | 75%以上 | 未測定 |
| **関数カバレッジ** | 90%以上 | 未測定 |

### 7.2 コンポーネント別目標

| コンポーネント | 優先度 | カバレッジ目標 |
|---------------|--------|---------------|
| `models.py` | 中 | 80% |
| `data_loader.py` | 高 | 85% |
| `data_transformer.py` | 高 | 85% |
| `similarity_calculator.py` | 高 | 80% |
| `recommendation_engine.py` | 最高 | 90% |
| `recommendation_system.py` | 高 | 85% |
| `scripts/` | 中 | 70% |

---

## 8. 実装優先度

### フェーズ1: 基礎テスト（1-2週間）

1. **データモデルテスト** (`test_models.py`) - 完全実装
2. **データローダーテスト** (`test_data_loader.py`) - コア機能
3. **データ変換テスト** (`test_data_transformer.py`) - レベル正規化、マスタ作成

### フェーズ2: コアロジックテスト（2-3週間）

4. **類似度計算テスト** (`test_similarity_calculator.py`) - Jaccard係数検証
5. **推薦エンジンテスト** (`test_recommendation_engine.py`) - スコア計算、推薦ロジック

### フェーズ3: 統合・E2Eテスト（1-2週間）

6. **推薦システムテスト** (`test_recommendation_system.py`) - ファサード機能
7. **統合テスト** (`test_integration.py`) - パイプライン全体
8. **E2Eテスト** (`test_e2e.py`) - スクリプト実行

### フェーズ4: エッジケース・パフォーマンス（1週間）

9. エッジケースの追加
10. パフォーマンステスト
11. ドキュメント整備

---

## 9. 付録

### 9.1 テストマーカー定義

```python
# pytest.ini または pyproject.toml に追加
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "e2e: marks tests as end-to-end tests",
    "unit: marks tests as unit tests",
]
```

### 9.2 参考リンク

- pytest 公式ドキュメント: https://docs.pytest.org/
- pytest-cov: https://pytest-cov.readthedocs.io/
- pandas テストガイド: https://pandas.pydata.org/docs/development/contributing_codetest.html

---

**改訂履歴**

| バージョン | 日付 | 変更内容 | 作成者 |
|-----------|------|----------|--------|
| 1.0 | 2025-10-23 | 初版作成 | Claude |
