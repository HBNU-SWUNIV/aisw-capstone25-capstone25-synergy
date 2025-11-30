import React, { useEffect, useMemo, useState } from 'react';
import { Button } from './ui/button';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { SpeakerStats } from '../types';
import { PieChart, Pie, Cell } from 'recharts';

const DEFAULT_SPEAKER_COLORS = ['#2563eb', '#16a34a', '#db2777', '#f59e0b', '#8b5cf6', '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#14b8a6'];

interface StatisticsProps {
  speakers: SpeakerStats[];
  avgTopic: number;
  avgNovelty: number;
  speakerColors: Record<string, string>;
}

export function Statistics({ speakers, avgTopic, avgNovelty, speakerColors }: StatisticsProps) {
  const [currentSpeaker, setCurrentSpeaker] = useState(0);
  const hasSpeakers = speakers.length > 0;

  useEffect(() => {
    if (!speakers.length) {
      setCurrentSpeaker(0);
      return;
    }
    if (currentSpeaker >= speakers.length) {
      setCurrentSpeaker(speakers.length - 1);
    }
  }, [speakers, currentSpeaker]);

  const colorMap = useMemo(() => speakerColors, [speakerColors]);

  const nextSpeaker = () => setCurrentSpeaker((prev) => (prev + 1) % Math.max(1, speakers.length));
  const prevSpeaker = () => setCurrentSpeaker((prev) => (prev - 1 + speakers.length) % Math.max(1, speakers.length));

  const currentData = speakers[currentSpeaker];

  const donutData = useMemo(
    () =>
      speakers.map((s) => ({
        key: s.speaker_id,
        value: Math.max(0, s.participation || 0),
        color: colorMap[s.speaker_id] ?? DEFAULT_SPEAKER_COLORS[0],
      })),
    [speakers, colorMap],
  );

  const renderPercentLabel = (props: any) => {
    const { percent, cx, cy, midAngle, innerRadius, outerRadius } = props;
    const RAD = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.55;
    const x = cx + radius * Math.cos(-midAngle * RAD);
    const y = cy + radius * Math.sin(-midAngle * RAD);
    const val = Math.round((percent || 0) * 100);
    if (val < 6) return null;
    return (
      <text x={x} y={y} fill="#374151" textAnchor="middle" dominantBaseline="central" fontSize={12}>
        {`${val}%`}
      </text>
    );
  };

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 shadow-sm h-full">
      <div className="grid grid-cols-2 gap-6 h-full">
        {/* LEFT: Individual Stats (Full Height) */}
        <div className="bg-white rounded-lg p-6 border border-gray-200 flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-[#374151]">개인 통계량</h3>
            {hasSpeakers && (
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={prevSpeaker} className="w-8 h-8 p-0 border-[#d1d1d1] bg-white">
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <span className="text-sm text-[#6b7280] w-12 text-center">{`${currentSpeaker + 1} / ${speakers.length}`}</span>
                <Button variant="outline" size="sm" onClick={nextSpeaker} className="w-8 h-8 p-0 border-[#d1d1d1] bg-white">
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            )}
          </div>

          {currentData ? (
            <div className="flex-1 flex items-end justify-center gap-12 px-8 pb-8">
              <div className="flex flex-col items-center justify-end gap-3 h-full">
                <div className="w-20 bg-[#e5e7eb] rounded-t-md relative" style={{ height: '240px' }}>
                  <div
                    className="absolute bottom-0 left-0 right-0 rounded-t-md"
                    style={{
                      height: `${(currentData.topic_avg / 10) * 100}%`,
                      backgroundColor: '#8b5cf6',
                    }}
                  />
                </div>
                <div className="text-base font-medium text-[#374151]">주제연관성</div>
              </div>
              <div className="flex flex-col items-center justify-end gap-3 h-full">
                <div className="w-20 bg-[#e5e7eb] rounded-t-md relative" style={{ height: '240px' }}>
                  <div
                    className="absolute bottom-0 left-0 right-0 rounded-t-md"
                    style={{
                      height: `${(currentData.novelty_avg / 10) * 100}%`,
                      backgroundColor: '#8b5cf6',
                    }}
                  />
                </div>
                <div className="text-base font-medium text-[#374151]">신규성</div>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-sm text-[#6b7280]">
              회의를 시작하면 개인 통계가 표시됩니다
            </div>
          )}
        </div>

        {/* RIGHT: Two stacked sections */}
        <div className="flex flex-col gap-6 h-full">
          {/* TOP RIGHT: Overall Stats */}
          <div className="bg-white rounded-lg p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-[#374151] mb-6">전체 통계</h3>
            <div className="space-y-6">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-[#374151]">평균 주제연관성</span>
                  <span className="text-sm font-semibold text-[#8b5cf6]">{avgTopic.toFixed(1)}</span>
                </div>
                <div className="w-full bg-white border border-gray-300 rounded-md h-7">
                  <div
                    className="h-full rounded-sm"
                    style={{
                      width: `${(avgTopic / 10) * 100}%`,
                      backgroundColor: '#e9d5ff',
                      backgroundImage: 'repeating-linear-gradient(-45deg, #c084fc, #c084fc 6px, #e9d5ff 6px, #e9d5ff 12px)',
                    }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-[#374151]">평균 신규성</span>
                  <span className="text-sm font-semibold text-[#8b5cf6]">{avgNovelty.toFixed(1)}</span>
                </div>
                <div className="w-full bg-white border border-gray-300 rounded-md h-7">
                  <div
                    className="h-full rounded-sm"
                    style={{
                      width: `${(avgNovelty / 10) * 100}%`,
                      backgroundColor: '#e9d5ff',
                      backgroundImage: 'repeating-linear-gradient(-45deg, #c084fc, #c084fc 6px, #e9d5ff 6px, #e9d5ff 12px)',
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* BOTTOM RIGHT: Donut Chart */}
          <div className="bg-white rounded-lg p-6 flex flex-col items-center justify-center border border-gray-200 flex-1">
            <h3 className="text-lg font-semibold text-[#374151] mb-4 self-start">화자별 점유율</h3>
            {speakers.length > 0 ? (
              <PieChart width={180} height={180}>
                <Pie
                  data={donutData}
                  dataKey="value"
                  nameKey="key"
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  label={renderPercentLabel}
                  isAnimationActive={false}
                >
                  {donutData.map((entry) => (
                    <Cell key={`cell-${entry.key}`} fill={entry.color} stroke="#f9fafb" strokeWidth={2} />
                  ))}
                </Pie>
              </PieChart>
            ) : (
              <div className="text-sm text-[#6b7280] flex-1 flex items-center justify-center">데이터 없음</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
