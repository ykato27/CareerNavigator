"""
Matrix Factorizationモデルのテスト
"""

import pytest
import pandas as pd
import numpy as np
from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel


# ==================== フィクスチャ ====================

@pytest.fixture
def sample_skill_matrix():
    """サンプル会員×力量マトリクス"""
    return pd.DataFrame(
        {
            's001': [3, 0, 2, 0, 1],
            's002': [0, 4, 0, 3, 0],
            's003': [2, 0, 5, 0, 2],
            's004': [0, 3, 0, 4, 0],
            's005': [1, 0, 2, 0, 3],
        },
        index=['m001', 'm002', 'm003', 'm004', 'm005']
    )


@pytest.fixture
def trained_model(sample_skill_matrix):
    """学習済みモデル"""
    model = MatrixFactorizationModel(n_components=2, random_state=42)
    model.fit(sample_skill_matrix)
    return model


# ==================== モデル初期化テスト ====================

class TestModelInitialization:
    """モデル初期化のテスト"""

    def test_initialization_default(self):
        """デフォルト初期化"""
        model = MatrixFactorizationModel()

        assert model.n_components == 20
        assert model.random_state == 42
        assert not model.is_fitted

    def test_initialization_custom_params(self):
        """カスタムパラメータで初期化"""
        model = MatrixFactorizationModel(
            n_components=10,
            random_state=123,
            max_iter=200
        )

        assert model.n_components == 10
        assert model.random_state == 123
        assert 'max_iter' in model.nmf_params
        assert model.nmf_params['max_iter'] == 200


# ==================== モデル学習テスト ====================

class TestModelFitting:
    """モデル学習のテスト"""

    def test_fit_basic(self, sample_skill_matrix):
        """基本的な学習"""
        model = MatrixFactorizationModel(n_components=2, random_state=42)
        model.fit(sample_skill_matrix)

        assert model.is_fitted
        assert model.W is not None
        assert model.H is not None
        assert model.W.shape == (5, 2)  # 5会員 × 2因子
        assert model.H.shape == (2, 5)  # 2因子 × 5力量

    def test_fit_preserves_codes(self, sample_skill_matrix):
        """学習後に会員・力量コードが保存される"""
        model = MatrixFactorizationModel(n_components=2)
        model.fit(sample_skill_matrix)

        assert model.member_codes == ['m001', 'm002', 'm003', 'm004', 'm005']
        assert model.competence_codes == ['s001', 's002', 's003', 's004', 's005']

    def test_fit_creates_index_mapping(self, sample_skill_matrix):
        """学習後にインデックスマッピングが作成される"""
        model = MatrixFactorizationModel(n_components=2)
        model.fit(sample_skill_matrix)

        assert model.member_index['m001'] == 0
        assert model.member_index['m003'] == 2
        assert model.competence_index['s001'] == 0
        assert model.competence_index['s003'] == 2

    def test_fit_reconstruction(self, sample_skill_matrix):
        """学習後の再構成が元に近い"""
        model = MatrixFactorizationModel(n_components=2, random_state=42)
        model.fit(sample_skill_matrix)

        # 再構成
        reconstructed = model.W @ model.H

        # 元データとの差が小さいことを確認（完全一致は不要）
        reconstruction_error = np.mean(np.abs(sample_skill_matrix.values - reconstructed))
        assert reconstruction_error < 1.5  # 平均誤差が1.5未満


# ==================== 予測テスト ====================

class TestPrediction:
    """予測のテスト"""

    def test_predict_all_competences(self, trained_model):
        """全力量に対する予測"""
        scores = trained_model.predict('m001')

        assert isinstance(scores, pd.Series)
        assert len(scores) == 5
        assert all(scores.index == ['s001', 's002', 's003', 's004', 's005'])

    def test_predict_specific_competences(self, trained_model):
        """特定の力量のみ予測"""
        scores = trained_model.predict('m001', competence_codes=['s001', 's003'])

        assert len(scores) == 2
        assert 's001' in scores.index
        assert 's003' in scores.index

    def test_predict_scores_are_non_negative(self, trained_model):
        """予測スコアが非負"""
        scores = trained_model.predict('m001')

        assert all(scores >= 0)

    def test_predict_unfitted_model_raises_error(self, sample_skill_matrix):
        """未学習モデルで予測するとエラー"""
        model = MatrixFactorizationModel()

        with pytest.raises(ValueError, match="モデルが学習されていません"):
            model.predict('m001')

    def test_predict_unknown_member_raises_error(self, trained_model):
        """未知の会員で予測するとエラー"""
        with pytest.raises(ValueError, match="学習データに存在しません"):
            trained_model.predict('m999')


# ==================== Top-K推薦テスト ====================

class TestTopKPrediction:
    """Top-K推薦のテスト"""

    def test_predict_top_k_basic(self, trained_model):
        """基本的なTop-K推薦"""
        top_k = trained_model.predict_top_k('m001', k=3, exclude_acquired=False)

        assert len(top_k) == 3
        assert all(isinstance(item, tuple) for item in top_k)
        assert all(len(item) == 2 for item in top_k)

        # スコア降順であることを確認
        scores = [score for _, score in top_k]
        assert scores == sorted(scores, reverse=True)

    def test_predict_top_k_exclude_acquired(self, trained_model, sample_skill_matrix):
        """既習得力量を除外したTop-K推薦"""
        # m001が習得している力量（スコア>0）を取得
        acquired = sample_skill_matrix.loc['m001']
        acquired_competences = acquired[acquired > 0].index.tolist()

        top_k = trained_model.predict_top_k(
            'm001',
            k=3,
            exclude_acquired=True,
            acquired_competences=acquired_competences
        )

        # 推薦に既習得力量が含まれていないことを確認
        recommended_codes = [code for code, _ in top_k]
        assert all(code not in acquired_competences for code in recommended_codes)

    def test_predict_top_k_without_acquired_param_raises_error(self, trained_model):
        """exclude_acquired=Trueで acquired_competences未指定はエラー"""
        with pytest.raises(ValueError, match="acquired_competencesを指定してください"):
            trained_model.predict_top_k('m001', k=3, exclude_acquired=True)


# ==================== 因子取得テスト ====================

class TestFactorRetrieval:
    """因子取得のテスト"""

    def test_get_member_factors(self, trained_model):
        """会員の潜在因子を取得"""
        factors = trained_model.get_member_factors('m001')

        assert isinstance(factors, np.ndarray)
        assert factors.shape == (2,)  # n_components=2
        assert all(factors >= 0)  # NMFは非負

    def test_get_competence_factors(self, trained_model):
        """力量の潜在因子を取得"""
        factors = trained_model.get_competence_factors('s001')

        assert isinstance(factors, np.ndarray)
        assert factors.shape == (2,)
        assert all(factors >= 0)

    def test_get_factors_unfitted_raises_error(self):
        """未学習モデルで因子取得するとエラー"""
        model = MatrixFactorizationModel()

        with pytest.raises(ValueError, match="モデルが学習されていません"):
            model.get_member_factors('m001')

    def test_get_factors_unknown_code_raises_error(self, trained_model):
        """未知のコードで因子取得するとエラー"""
        with pytest.raises(ValueError, match="学習データに存在しません"):
            trained_model.get_member_factors('m999')


# ==================== モデル保存・読み込みテスト ====================

class TestModelPersistence:
    """モデル保存・読み込みのテスト"""

    def test_save_and_load(self, trained_model, tmp_path):
        """モデルの保存と読み込み"""
        filepath = tmp_path / 'model.pkl'

        # 保存
        trained_model.save(str(filepath))
        assert filepath.exists()

        # 読み込み
        loaded_model = MatrixFactorizationModel.load(str(filepath))

        # 同じ結果を返すことを確認
        original_scores = trained_model.predict('m001')
        loaded_scores = loaded_model.predict('m001')

        pd.testing.assert_series_equal(original_scores, loaded_scores)

    def test_save_unfitted_raises_error(self, tmp_path):
        """未学習モデルの保存はエラー"""
        model = MatrixFactorizationModel()
        filepath = tmp_path / 'model.pkl'

        with pytest.raises(ValueError, match="モデルが学習されていません"):
            model.save(str(filepath))

    def test_loaded_model_is_fitted(self, trained_model, tmp_path):
        """読み込まれたモデルは学習済み状態"""
        filepath = tmp_path / 'model.pkl'
        trained_model.save(str(filepath))

        loaded_model = MatrixFactorizationModel.load(str(filepath))

        assert loaded_model.is_fitted
        assert loaded_model.member_codes == trained_model.member_codes
        assert loaded_model.competence_codes == trained_model.competence_codes


# ==================== エッジケーステスト ====================

class TestEdgeCases:
    """エッジケースのテスト"""

    def test_small_matrix(self):
        """小さなマトリクスでの学習"""
        small_matrix = pd.DataFrame(
            {'s001': [1, 0], 's002': [0, 1]},
            index=['m001', 'm002']
        )

        model = MatrixFactorizationModel(n_components=1, random_state=42)
        model.fit(small_matrix)

        assert model.is_fitted
        scores = model.predict('m001')
        assert len(scores) == 2

    def test_sparse_matrix(self):
        """疎なマトリクスでの学習"""
        sparse_matrix = pd.DataFrame(
            np.zeros((10, 10)),
            index=[f'm{i:03d}' for i in range(10)],
            columns=[f's{i:03d}' for i in range(10)]
        )
        # 少しだけ値を設定
        sparse_matrix.iloc[0, 0] = 3
        sparse_matrix.iloc[1, 1] = 4
        sparse_matrix.iloc[2, 2] = 2

        model = MatrixFactorizationModel(n_components=3, random_state=42)
        model.fit(sparse_matrix)

        assert model.is_fitted

    def test_reconstruction_error(self, trained_model):
        """再構成誤差の取得"""
        error = trained_model.get_reconstruction_error()

        assert isinstance(error, (int, float))
        assert error >= 0


# ==================== 統合テスト ====================

class TestIntegration:
    """統合テスト"""

    def test_full_workflow(self, sample_skill_matrix, tmp_path):
        """完全なワークフロー"""
        # 1. モデル初期化
        model = MatrixFactorizationModel(n_components=2, random_state=42)

        # 2. 学習
        model.fit(sample_skill_matrix)

        # 3. 予測
        scores = model.predict('m001')
        assert len(scores) == 5

        # 4. Top-K推薦
        acquired = sample_skill_matrix.loc['m001']
        acquired_competences = acquired[acquired > 0].index.tolist()
        top_k = model.predict_top_k(
            'm001',
            k=2,
            exclude_acquired=True,
            acquired_competences=acquired_competences
        )
        assert len(top_k) == 2

        # 5. 因子取得
        member_factors = model.get_member_factors('m001')
        assert member_factors.shape == (2,)

        # 6. 保存
        filepath = tmp_path / 'model.pkl'
        model.save(str(filepath))

        # 7. 読み込み
        loaded_model = MatrixFactorizationModel.load(str(filepath))
        loaded_scores = loaded_model.predict('m001')

        pd.testing.assert_series_equal(scores, loaded_scores)
