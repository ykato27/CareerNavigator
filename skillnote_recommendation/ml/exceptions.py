"""
機械学習推薦システム用のカスタム例外
"""


class ColdStartError(Exception):
    """
    コールドスタート問題によるエラー

    新規メンバーなど、学習データに存在しないメンバーに対して推薦を試みた場合に発生します。
    """

    def __init__(self, member_code: str, message: str = None):
        self.member_code = member_code
        if message is None:
            message = (
                f"メンバーコード '{member_code}' の保有力量が登録されていないため、ML推薦ができません。\n"
                f"【コールドスタート問題】\n"
                f"このメンバーの力量データを登録してから、再度ML推薦を実行してください。"
            )
        super().__init__(message)


class MLModelNotTrainedError(Exception):
    """MLモデルが未学習のエラー"""

    def __init__(self, message: str = "MLモデルが学習されていません。先に学習を実行してください。"):
        super().__init__(message)
