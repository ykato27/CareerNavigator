"""
推薦システム評価

時系列分割による評価とメトリクス計算を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from skillnote_recommendation.core.recommendation_engine import RecommendationEngine


class RecommendationEvaluator:
    """推薦システム評価クラス"""

    def __init__(self, engine: RecommendationEngine = None):
        """
        初期化

        Args:
            engine: 推薦エンジン（オプション）
        """
        self.engine = engine

    def temporal_train_test_split(
        self,
        member_competence: pd.DataFrame,
        split_date: str = None,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        時系列による学習・評価データ分割

        Args:
            member_competence: 会員習得力量データ（取得日カラム必須）
            split_date: 分割日（YYYY-MM-DD形式、Noneの場合はtrain_ratioで自動計算）
            train_ratio: 学習データ割合（split_dateがNoneの場合に使用）

        Returns:
            (学習データ, 評価データ)のタプル

        Raises:
            ValueError: 取得日カラムがない場合
        """
        # 取得日カラムの確認
        if '取得日' not in member_competence.columns:
            raise ValueError("会員習得力量データに '取得日' カラムが必要です")

        # 取得日をdatetime型に変換
        df = member_competence.copy()
        df['取得日_dt'] = pd.to_datetime(df['取得日'], errors='coerce')

        # 欠損値を除外
        df = df[df['取得日_dt'].notna()].copy()

        if len(df) == 0:
            raise ValueError("有効な取得日を持つデータがありません")

        # 分割日の決定
        if split_date is None:
            # データを時系列でソートし、train_ratio位置で分割
            sorted_dates = df['取得日_dt'].sort_values()
            split_idx = int(len(sorted_dates) * train_ratio)
            split_datetime = sorted_dates.iloc[split_idx]
        else:
            split_datetime = pd.to_datetime(split_date)

        # 学習データと評価データに分割
        train_data = df[df['取得日_dt'] < split_datetime].copy()
        test_data = df[df['取得日_dt'] >= split_datetime].copy()

        # 一時カラムを削除
        train_data = train_data.drop(columns=['取得日_dt'])
        test_data = test_data.drop(columns=['取得日_dt'])

        return train_data, test_data

    def evaluate_recommendations(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        competence_master: pd.DataFrame,
        top_k: int = 10,
        member_sample: Optional[List[str]] = None,
        similarity_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        推薦結果を評価

        Args:
            train_data: 学習データ（過去の習得力量）
            test_data: 評価データ（将来の習得力量）
            competence_master: 力量マスタ
            top_k: 推薦する上位K件
            member_sample: 評価対象会員リスト（Noneの場合は全会員）
            similarity_data: 類似度データ（Noneの場合は空のDataFrame）

        Returns:
            評価メトリクスの辞書
        """
        # 評価対象会員の決定
        if member_sample is None:
            # テストデータで習得記録がある会員
            member_sample = test_data['メンバーコード'].unique().tolist()

        # エンジンの準備
        if self.engine is None:
            # 類似度データの準備
            if similarity_data is None:
                similarity_data = pd.DataFrame(columns=['力量1', '力量2', '類似度'])

            # 会員マスタの準備（メンバーコードのみ）
            members_data = pd.DataFrame({
                'メンバーコード': train_data['メンバーコード'].unique()
            })

            # 推薦エンジンを作成
            engine = RecommendationEngine(
                df_members=members_data,
                df_competence_master=competence_master,
                df_member_competence=train_data,
                df_similarity=similarity_data
            )
        else:
            engine = self.engine

        # メトリクス集計用
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        hit_count = 0
        total_members = 0

        for member_code in member_sample:
            # テストデータでの習得力量（正解データ）
            actual_acquired = test_data[
                test_data['メンバーコード'] == member_code
            ]['力量コード'].unique().tolist()

            if len(actual_acquired) == 0:
                continue

            total_members += 1

            # 学習データを使って推薦を生成
            recommendations = engine.recommend(
                member_code=member_code,
                top_n=top_k
            )

            # 推薦された力量コード
            recommended_codes = [rec.competence_code for rec in recommendations]

            if len(recommended_codes) == 0:
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                ndcg_scores.append(0.0)
                continue

            # Precision@K: 推薦のうち実際に習得した割合
            hits = len(set(recommended_codes) & set(actual_acquired))
            precision = hits / len(recommended_codes) if len(recommended_codes) > 0 else 0.0
            precision_scores.append(precision)

            # Recall@K: 実際に習得したもののうち推薦に含まれていた割合
            recall = hits / len(actual_acquired) if len(actual_acquired) > 0 else 0.0
            recall_scores.append(recall)

            # NDCG@K: ランキングの質を評価
            ndcg = self._calculate_ndcg(recommended_codes, actual_acquired, top_k)
            ndcg_scores.append(ndcg)

            # Hit: 少なくとも1つ正解があったか
            if hits > 0:
                hit_count += 1

        # メトリクスの平均を計算
        if total_members == 0:
            return {
                f'precision@{top_k}': 0.0,
                f'recall@{top_k}': 0.0,
                f'ndcg@{top_k}': 0.0,
                'hit_rate': 0.0,
                'evaluated_members': 0
            }

        return {
            f'precision@{top_k}': np.mean(precision_scores) if precision_scores else 0.0,
            f'recall@{top_k}': np.mean(recall_scores) if recall_scores else 0.0,
            f'ndcg@{top_k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'hit_rate': hit_count / total_members if total_members > 0 else 0.0,
            'evaluated_members': total_members
        }

    def _calculate_ndcg(
        self,
        recommended_items: List[str],
        relevant_items: List[str],
        k: int
    ) -> float:
        """
        NDCG@K（Normalized Discounted Cumulative Gain）を計算

        Args:
            recommended_items: 推薦アイテムリスト（ランキング順）
            relevant_items: 関連アイテムリスト（正解）
            k: 上位K件

        Returns:
            NDCG@K スコア（0.0-1.0）
        """
        # DCG（Discounted Cumulative Gain）を計算
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                # 順位が高いほど重要（対数で割引）
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

        # IDCG（Ideal DCG）を計算
        # 全ての関連アイテムが上位にランクされた場合のDCG
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)

        # NDCG = DCG / IDCG
        if idcg == 0:
            return 0.0

        return dcg / idcg

    def cross_validate_temporal(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        n_splits: int = 5,
        top_k: int = 10
    ) -> List[Dict[str, float]]:
        """
        時系列クロスバリデーション

        データを時系列で複数のfoldに分割し、各foldで評価を実行

        Args:
            member_competence: 会員習得力量データ
            competence_master: 力量マスタ
            n_splits: 分割数
            top_k: 推薦する上位K件

        Returns:
            各foldの評価メトリクスリスト
        """
        # 取得日でソート
        df = member_competence.copy()
        df['取得日_dt'] = pd.to_datetime(df['取得日'], errors='coerce')
        df = df[df['取得日_dt'].notna()].sort_values('取得日_dt')
        df = df.drop(columns=['取得日_dt'])

        # データサイズ
        total_size = len(df)
        fold_size = total_size // (n_splits + 1)

        results = []

        for i in range(n_splits):
            # 累積的に学習データを増やし、次のfoldをテストデータとする
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = test_start + fold_size

            train_data = df.iloc[:train_end]
            test_data = df.iloc[test_start:test_end]

            if len(train_data) == 0 or len(test_data) == 0:
                continue

            # 評価実行
            metrics = self.evaluate_recommendations(
                train_data=train_data,
                test_data=test_data,
                competence_master=competence_master,
                top_k=top_k
            )

            metrics['fold'] = i + 1
            metrics['train_size'] = len(train_data)
            metrics['test_size'] = len(test_data)

            results.append(metrics)

        return results

    def print_evaluation_results(self, metrics: Dict[str, float]):
        """
        評価結果を表示

        Args:
            metrics: 評価メトリクスの辞書
        """
        print("\n" + "=" * 80)
        print("推薦システム評価結果")
        print("=" * 80)

        print(f"\n評価対象会員数: {metrics.get('evaluated_members', 0)}名")

        # K値を取得
        k = None
        for key in metrics.keys():
            if key.startswith('precision@'):
                k = key.split('@')[1]
                break

        if k:
            print(f"\n【Top-{k} 推薦の評価】")
            print(f"  Precision@{k}: {metrics.get(f'precision@{k}', 0.0):.4f}")
            print(f"  Recall@{k}:    {metrics.get(f'recall@{k}', 0.0):.4f}")
            print(f"  NDCG@{k}:      {metrics.get(f'ndcg@{k}', 0.0):.4f}")
            print(f"  Hit Rate:      {metrics.get('hit_rate', 0.0):.4f}")

        print("\n" + "=" * 80)

    def export_evaluation_results(
        self,
        metrics: Dict[str, float],
        output_path: str
    ):
        """
        評価結果をCSVに出力

        Args:
            metrics: 評価メトリクスの辞書
            output_path: 出力ファイルパス
        """
        df = pd.DataFrame([metrics])
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n評価結果を出力: {output_path}")
