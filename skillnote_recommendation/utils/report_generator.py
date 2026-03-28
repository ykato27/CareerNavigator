"""
SEM分析レポート生成ユーティリティ

HTMLレポートを生成し、ブラウザでPDF保存可能な形式で提供します。
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime


def generate_html_report(
    member_code: str,
    member_name: str,
    member_info: Dict[str, Any],
    domain_scores: Dict[str, float],
    recommendations: List[Any],
    gaps_by_domain: Dict[str, List[Dict[str, Any]]],
    fit_indices: Optional[Dict[str, Dict[str, float]]] = None
) -> str:
    """
    SEM分析のHTMLレポートを生成

    Args:
        member_code: メンバーコード
        member_name: メンバー名
        member_info: メンバー情報
        domain_scores: 領域別スコア
        recommendations: 推薦リスト
        gaps_by_domain: 領域別ギャップ情報
        fit_indices: モデル適合度指標（領域別）

    Returns:
        HTMLレポート文字列
    """
    # 現在日時
    report_date = datetime.now().strftime("%Y年%m月%d日 %H:%M")

    # HTMLヘッダー
    html = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SEM分析レポート - {member_name}</title>
        <style>
            @media print {{
                body {{
                    -webkit-print-color-adjust: exact;
                    print-color-adjust: exact;
                }}
                .page-break {{
                    page-break-before: always;
                }}
            }}

            body {{
                font-family: 'Yu Gothic', 'Meiryo', sans-serif;
                margin: 40px;
                background: #ffffff;
                color: #333;
            }}

            h1 {{
                color: #1f77b4;
                border-bottom: 3px solid #1f77b4;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}

            h2 {{
                color: #2e7d32;
                border-bottom: 2px solid #2e7d32;
                padding-bottom: 5px;
                margin-top: 30px;
                margin-bottom: 15px;
            }}

            h3 {{
                color: #555;
                margin-top: 20px;
                margin-bottom: 10px;
            }}

            .header-info {{
                background: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 30px;
            }}

            .header-info p {{
                margin: 5px 0;
            }}

            .metric-card {{
                display: inline-block;
                background: #e3f2fd;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 10px 10px 0;
                min-width: 150px;
            }}

            .metric-card h4 {{
                margin: 0 0 5px 0;
                color: #1976d2;
                font-size: 14px;
            }}

            .metric-card .value {{
                font-size: 24px;
                font-weight: bold;
                color: #0d47a1;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
            }}

            th {{
                background: #1f77b4;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }}

            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }}

            tr:hover {{
                background: #f5f5f5;
            }}

            .badge-acquired {{
                background: #4caf50;
                color: white;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 12px;
            }}

            .badge-not-acquired {{
                background: #f44336;
                color: white;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 12px;
            }}

            .badge-significant {{
                background: #2e7d32;
                color: white;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 12px;
            }}

            .recommendation {{
                background: #fff3e0;
                border-left: 4px solid #ff9800;
                padding: 15px;
                margin: 10px 0;
            }}

            .recommendation h4 {{
                margin: 0 0 5px 0;
                color: #e65100;
            }}

            .footer {{
                margin-top: 50px;
                text-align: center;
                color: #999;
                font-size: 12px;
                border-top: 1px solid #ddd;
                padding-top: 20px;
            }}

            .domain-section {{
                margin: 20px 0;
                padding: 15px;
                background: #fafafa;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>🔬 SEM分析レポート</h1>

        <div class="header-info">
            <p><strong>作成日時:</strong> {report_date}</p>
            <p><strong>メンバー:</strong> {member_name} ({member_code})</p>
            <p><strong>職種:</strong> {member_info.get('職種', 'N/A')}</p>
            <p><strong>役職:</strong> {member_info.get('役職名', 'N/A')}</p>
            <p><strong>職能等級:</strong> {member_info.get('職能等級', 'N/A')}</p>
        </div>

        <h2>📊 領域別習得度プロファイル</h2>
        <div>
    """

    # 領域別スコアを表示
    for domain, score in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True):
        html += f"""
            <div class="metric-card">
                <h4>{domain}</h4>
                <div class="value">{score*100:.1f}%</div>
            </div>
        """

    html += """
        </div>
        <div class="page-break"></div>

        <h2>🎯 推薦される力量（上位10件）</h2>
    """

    # 推薦を表示
    for i, rec in enumerate(recommendations[:10], 1):
        html += f"""
        <div class="recommendation">
            <h4>#{i} {rec.competence_name}</h4>
            <p><strong>タイプ:</strong> {rec.competence_type} |
               <strong>領域:</strong> {rec.domain} |
               <strong>SEMスコア:</strong> {rec.sem_score:.3f}</p>
            <p><strong>現在レベル:</strong> {rec.current_level} →
               <strong>目標レベル:</strong> {rec.target_level}</p>
            <p><strong>推薦理由:</strong> {rec.reason}</p>
            {f'<p><strong>パス係数:</strong> {rec.path_coefficient:.3f} ' + ('<span class="badge-significant">有意</span>' if rec.is_significant else '') + '</p>' if rec.path_coefficient else ''}
        </div>
        """

    html += """
        <div class="page-break"></div>
        <h2>✅ 持っている力量 / ❌ 持っていない力量</h2>
    """

    # ギャップ分析を表示
    for domain, gap_list in gaps_by_domain.items():
        acquired = [g for g in gap_list if g['is_acquired']]
        not_acquired = [g for g in gap_list if not g['is_acquired']]

        html += f"""
        <div class="domain-section">
            <h3>📂 {domain} 領域</h3>
            <p><strong>習得済み:</strong> {len(acquired)}件 | <strong>未習得:</strong> {len(not_acquired)}件</p>

            <h4>✅ 習得済みの力量</h4>
            <table>
                <thead>
                    <tr>
                        <th>力量名</th>
                        <th>タイプ</th>
                        <th>レベル</th>
                    </tr>
                </thead>
                <tbody>
        """

        for comp in acquired[:10]:  # 最大10件
            html += f"""
                    <tr>
                        <td>{comp['competence_name']}</td>
                        <td>{comp['competence_type']}</td>
                        <td>{comp.get('level', 'N/A')}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>

            <h4>❌ 未習得の力量</h4>
            <table>
                <thead>
                    <tr>
                        <th>力量名</th>
                        <th>タイプ</th>
                    </tr>
                </thead>
                <tbody>
        """

        for comp in not_acquired[:10]:  # 最大10件
            html += f"""
                    <tr>
                        <td>{comp['competence_name']}</td>
                        <td>{comp['competence_type']}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """

    # モデル適合度指標を表示
    if fit_indices:
        html += """
        <div class="page-break"></div>
        <h2>📈 モデル適合度指標</h2>
        """

        for domain, indices in fit_indices.items():
            html += f"""
            <div class="domain-section">
                <h3>{domain} 領域</h3>
                <div>
                    <div class="metric-card">
                        <h4>GFI (適合度)</h4>
                        <div class="value">{indices['gfi']:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>NFI (規準適合度)</h4>
                        <div class="value">{indices['nfi']:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>説明分散 (R²)</h4>
                        <div class="value">{indices['variance_explained']:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>平均効果サイズ</h4>
                        <div class="value">{indices['avg_effect_size']:.3f}</div>
                    </div>
                </div>
            </div>
            """

    # フッター
    html += f"""
        <div class="footer">
            <p>© 2025 CareerNavigator - SEM分析レポート</p>
            <p>このレポートは構造方程式モデリング（SEM）に基づいて生成されました。</p>
        </div>
    </body>
    </html>
    """

    return html
