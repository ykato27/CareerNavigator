import React from 'react';

export const Dashboard = () => {
    return (
        <div className="space-y-6">
            <div className="text-center py-12">
                <h2 className="text-2xl font-bold text-gray-700 mb-2">メンバーの力量をシートで管理してみましょう。</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Card 1 */}
                <div className="bg-white p-8 rounded-2xl shadow-sm hover:shadow-md transition-shadow flex flex-col items-center text-center">
                    <div className="w-16 h-16 bg-green-100 rounded-lg flex items-center justify-center mb-4 text-green-600">
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><polyline points="9 22 9 12 15 12 15 22" /></svg>
                    </div>
                    <h3 className="text-lg font-bold text-gray-800 mb-2">プロジェクトを設定</h3>
                    <p className="text-sm text-gray-500 leading-relaxed">
                        企業や工場、部署など、管理したい組織単位でプロジェクトを作成してみましょう。
                    </p>
                </div>

                {/* Card 2 */}
                <div className="bg-white p-8 rounded-2xl shadow-sm hover:shadow-md transition-shadow flex flex-col items-center text-center">
                    <div className="w-16 h-16 bg-green-100 rounded-lg flex items-center justify-center mb-4 text-green-600">
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" /><path d="M3 9h18" /><path d="M9 21V9" /></svg>
                    </div>
                    <h3 className="text-lg font-bold text-gray-800 mb-2">シートを作成</h3>
                    <p className="text-sm text-gray-500 leading-relaxed">
                        メンバーの力量マップや育成計画、個人力量を管理するシートを作成できます。運用にそってシートを使い分けてみてください。
                    </p>
                </div>

                {/* Card 3 */}
                <div className="bg-white p-8 rounded-2xl shadow-sm hover:shadow-md transition-shadow flex flex-col items-center text-center">
                    <div className="w-16 h-16 bg-green-100 rounded-lg flex items-center justify-center mb-4 text-green-600">
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" /><path d="m15 5 4 4" /></svg>
                    </div>
                    <h3 className="text-lg font-bold text-gray-800 mb-2">運用開始</h3>
                    <p className="text-sm text-gray-500 leading-relaxed">
                        シートを作成したらメンバーにスキルや教育・資格を登録したり、育成計画の記録をつけてみましょう。
                    </p>
                </div>
            </div>
        </div>
    );
};
