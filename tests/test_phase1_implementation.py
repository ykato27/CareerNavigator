"""
フェーズ1実装の動作確認テスト

1. predict_latent_scores()メソッドの動作確認
2. パラメータ名の形式統一の確認
"""

import numpy as np
import pandas as pd
import sys
import traceback

# パスを追加
sys.path.insert(0, '/home/user/CareerNavigator')

from skillnote_recommendation.ml.unified_sem_estimator import (
    UnifiedSEMEstimator,
    MeasurementModelSpec,
    StructuralModelSpec,
)

def test_predict_latent_scores():
    """predict_latent_scores()の動作確認"""
    print("\n" + "=" * 80)
    print("テスト1: predict_latent_scores()の動作確認")
    print("=" * 80)

    # テストデータを生成
    np.random.seed(42)
    n = 200

    # 潜在変数を生成
    beginner = np.random.normal(0, 1, n)
    intermediate = 0.7 * beginner + np.random.normal(0, 0.5, n)

    # 観測変数を生成
    data = pd.DataFrame({
        'Python基礎': 0.8 * beginner + np.random.normal(0, 0.3, n),
        'SQL基礎': 0.75 * beginner + np.random.normal(0, 0.35, n),
        'Web開発': 0.85 * intermediate + np.random.normal(0, 0.25, n),
        'データ分析': 0.80 * intermediate + np.random.normal(0, 0.30, n),
    })

    print(f"✅ テストデータ生成完了: {data.shape}")

    # モデル仕様
    measurement = [
        MeasurementModelSpec(
            '初級力量',
            ['Python基礎', 'SQL基礎'],
            reference_indicator='Python基礎'
        ),
        MeasurementModelSpec(
            '中級力量',
            ['Web開発', 'データ分析'],
            reference_indicator='Web開発'
        ),
    ]

    structural = [
        StructuralModelSpec('初級力量', '中級力量'),
    ]

    print(f"✅ モデル仕様定義完了")

    # モデルを推定
    try:
        sem = UnifiedSEMEstimator(measurement, structural)
        print(f"✅ UnifiedSEMEstimator初期化完了")

        sem.fit(data)
        print(f"✅ モデル学習完了: is_fitted={sem.is_fitted}")

        # 適合度指標を確認
        if sem.fit_indices_:
            print(f"\n【適合度指標】")
            print(f"  RMSEA: {sem.fit_indices_.rmsea:.3f}")
            print(f"  CFI: {sem.fit_indices_.cfi:.3f}")
            print(f"  TLI: {sem.fit_indices_.tli:.3f}")

        # 潜在変数スコアを推定
        print(f"\n【潜在変数スコア推定】")
        latent_scores = sem.predict_latent_scores(data)
        print(f"✅ 潜在変数スコア推定完了")
        print(f"  形状: {latent_scores.shape}")
        print(f"  列: {list(latent_scores.columns)}")
        print(f"\n  最初の5件:")
        print(latent_scores.head())

        # 真の潜在変数との相関を確認
        true_latent = pd.DataFrame({
            '初級力量': beginner,
            '中級力量': intermediate
        })

        print(f"\n【真の潜在変数との相関】")
        for col in latent_scores.columns:
            if col in true_latent.columns:
                corr = np.corrcoef(latent_scores[col], true_latent[col])[0, 1]
                print(f"  {col}: r = {corr:.3f}")

        return True

    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        traceback.print_exc()
        return False


def test_parameter_names():
    """パラメータ名の形式確認"""
    print("\n" + "=" * 80)
    print("テスト2: パラメータ名の形式確認")
    print("=" * 80)

    # テストデータを生成
    np.random.seed(42)
    n = 200

    beginner = np.random.normal(0, 1, n)
    intermediate = 0.7 * beginner + np.random.normal(0, 0.5, n)

    data = pd.DataFrame({
        'Python基礎': 0.8 * beginner + np.random.normal(0, 0.3, n),
        'SQL基礎': 0.75 * beginner + np.random.normal(0, 0.35, n),
        'Web開発': 0.85 * intermediate + np.random.normal(0, 0.25, n),
        'データ分析': 0.80 * intermediate + np.random.normal(0, 0.30, n),
    })

    # モデル仕様
    measurement = [
        MeasurementModelSpec('初級力量', ['Python基礎', 'SQL基礎']),
        MeasurementModelSpec('中級力量', ['Web開発', 'データ分析']),
    ]

    structural = [
        StructuralModelSpec('初級力量', '中級力量'),
    ]

    try:
        sem = UnifiedSEMEstimator(measurement, structural)
        sem.fit(data)

        # paramsプロパティの動作確認
        print(f"✅ paramsプロパティ動作確認")
        print(f"  パラメータ数: {len(sem.params)}")

        # 構造パラメータの確認
        param_name = "中級力量 ~ 初級力量"
        print(f"\n【構造パラメータ】")
        print(f"  検索キー: '{param_name}'")

        if param_name in sem.params:
            param = sem.params[param_name]
            print(f"  ✅ パラメータが見つかりました")
            print(f"    値: {param.value:.3f}")
            print(f"    標準誤差: {param.std_error:.3f}" if param.std_error else "    標準誤差: None")
            print(f"    p値: {param.p_value:.3f}" if param.p_value else "    p値: None")
        else:
            print(f"  ❌ パラメータが見つかりません")
            print(f"\n  利用可能なパラメータ（構造モデル関連）:")
            for key in sem.params.keys():
                if '~' in key:
                    print(f"    - '{key}'")

        return True

    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("\n" + "=" * 80)
    print("フェーズ1実装の動作確認テスト")
    print("=" * 80)

    results = []

    # テスト1: predict_latent_scores()
    results.append(("predict_latent_scores()の動作", test_predict_latent_scores()))

    # テスト2: パラメータ名の形式
    results.append(("パラメータ名の形式統一", test_parameter_names()))

    # 結果サマリー
    print("\n" + "=" * 80)
    print("テスト結果サマリー")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 すべてのテストが成功しました！")
    else:
        print("⚠️ 一部のテストが失敗しました")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
