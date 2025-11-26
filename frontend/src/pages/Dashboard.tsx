import { Home, LayoutGrid, Edit } from 'lucide-react';

export const Dashboard = () => {
    return (
        <div className="flex-1 px-8 py-12">
            <div className="text-center mb-16">
                <h2 className="text-2xl font-normal text-gray-700">メンバーの力量を</h2>
                <h2 className="text-2xl font-normal text-gray-700 mb-2">シートで管理してみましょう。</h2>
            </div>

            <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
                {/* Card 1 */}
                <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow">
                    <div className="p-8 flex flex-col items-center text-center">
                        <div className="w-24 h-24 bg-gray-50 rounded-lg flex items-center justify-center mb-6">
                            <Home size={48} className="text-gray-300" strokeWidth={1} />
                        </div>
                        <h3 className="text-base font-bold text-gray-800 mb-4">プロジェクトを<br />設定</h3>
                        <p className="text-sm text-gray-600 leading-relaxed">
                            企業や工場、部署など、<br />
                            管理したい組織単位でプ<br />
                            ロジェクトを作成してみ<br />
                            ましょう。
                        </p>
                    </div>
                </div>

                {/* Card 2 */}
                <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow">
                    <div className="p-8 flex flex-col items-center text-center">
                        <div className="w-24 h-24 bg-gray-50 rounded-lg flex items-center justify-center mb-6">
                            <LayoutGrid size={48} className="text-gray-300" strokeWidth={1} />
                        </div>
                        <h3 className="text-base font-bold text-gray-800 mb-4">シートを作成</h3>
                        <p className="text-sm text-gray-600 leading-relaxed">
                            メンバーの力量マップや<br />
                            育成計画、個人力量を管<br />
                            理するシートを作成でき<br />
                            ます。運用にそってシー<br />
                            トを使い分けてみてくだ<br />
                            さい。
                        </p>
                    </div>
                </div>

                {/* Card 3 */}
                <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow">
                    <div className="p-8 flex flex-col items-center text-center">
                        <div className="w-24 h-24 bg-gray-50 rounded-lg flex items-center justify-center mb-6">
                            <Edit size={48} className="text-gray-300" strokeWidth={1} />
                        </div>
                        <h3 className="text-base font-bold text-gray-800 mb-4">運用開始</h3>
                        <p className="text-sm text-gray-600 leading-relaxed">
                            シートを作成したらメン<br />
                            バーにスキルや教育・資<br />
                            格を登録したり、育成計<br />
                            画の記録をつけてみまし<br />
                            ょう。
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};
