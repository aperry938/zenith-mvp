import React from 'react';
import { InfoTooltip } from './InfoTooltip';

interface BioDeviation {
    feature: string;
    feature_idx: number;
    value: number;
    ideal_lo: number;
    ideal_hi: number;
    deviation: number;
    direction: 'above' | 'below';
}

interface BiomechanicalPanelProps {
    bioQuality: number | null | undefined;
    bioDeviations: BioDeviation[] | null | undefined;
    bioFeatures: number[] | null | undefined;
    poseLabel: string | null | undefined;
}

const FEATURE_LABELS: Record<string, string> = {
    l_shoulder_flexion: 'L Shoulder',
    r_shoulder_flexion: 'R Shoulder',
    l_shoulder_abduction: 'L Sh Abd',
    r_shoulder_abduction: 'R Sh Abd',
    l_elbow_flexion: 'L Elbow',
    r_elbow_flexion: 'R Elbow',
    l_hip_flexion: 'L Hip',
    r_hip_flexion: 'R Hip',
    l_hip_abduction: 'L Hip Abd',
    r_hip_abduction: 'R Hip Abd',
    l_knee_flexion: 'L Knee',
    r_knee_flexion: 'R Knee',
    l_ankle_dorsiflexion: 'L Ankle',
    r_ankle_dorsiflexion: 'R Ankle',
    spinal_lateral_flexion: 'Lat Flex',
    trunk_forward_lean: 'Fwd Lean',
    shoulder_angle_symmetry: 'Sh Symm',
    elbow_angle_symmetry: 'Elb Symm',
    hip_angle_symmetry: 'Hip Symm',
    knee_angle_symmetry: 'Knee Symm',
};

function qualityColor(q: number): string {
    if (q >= 90) return 'text-green-400';
    if (q >= 70) return 'text-yellow-400';
    if (q >= 50) return 'text-orange-400';
    return 'text-red-400';
}

function qualityBarColor(q: number): string {
    if (q >= 90) return 'bg-green-500';
    if (q >= 70) return 'bg-yellow-500';
    if (q >= 50) return 'bg-orange-500';
    return 'bg-red-500';
}

function deviationIcon(direction: string): string {
    return direction === 'above' ? '↑' : '↓';
}

export const BiomechanicalPanel: React.FC<BiomechanicalPanelProps> = ({
    bioQuality,
    bioDeviations,
    bioFeatures,
    poseLabel: _poseLabel,
}) => {
    const q = bioQuality ?? 0;

    return (
        <div className="absolute top-5 left-5 w-64 flex flex-col gap-3 pointer-events-none">
            {/* Bio Quality Score */}
            <div className="relative z-20 bg-zenith-panel/85 border border-zinc-800 p-4 rounded-lg backdrop-blur-sm pointer-events-auto">
                <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-1 flex items-center">
                    BIO QUALITY
                    <InfoTooltip text="Biomechanical quality score (0-100). Computed from 30 expert-designed features (joint angles, symmetry, stability) compared against pose-specific ideal ranges defined by a kinesiologist." />
                </h3>
                <div className={`text-3xl font-bold font-mono ${qualityColor(q)} drop-shadow-[0_0_10px_rgba(0,255,153,0.3)]`}>
                    {bioQuality != null ? bioQuality.toFixed(0) : '--'}
                </div>
                {/* Quality bar */}
                <div className="mt-2 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                        className={`h-full rounded-full transition-all duration-300 ${qualityBarColor(q)}`}
                        style={{ width: `${Math.min(100, q)}%` }}
                    />
                </div>
            </div>

            {/* Deviations */}
            {bioDeviations && bioDeviations.length > 0 && (
                <div className="relative bg-zenith-panel/85 border border-zinc-800 p-3 rounded-lg backdrop-blur-sm pointer-events-auto z-20">
                    <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-2 flex items-center">
                        CORRECTIONS
                        <InfoTooltip text="Top 3 biomechanical deviations from ideal alignment. Arrows show direction (above/below ideal range). Values show your measurement vs. the ideal range for this pose." />
                    </h3>
                    <div className="flex flex-col gap-1.5">
                        {bioDeviations.slice(0, 3).map((dev, i) => {
                            const label = FEATURE_LABELS[dev.feature] || dev.feature;
                            const isAngle = dev.feature_idx < 16;
                            const unit = isAngle ? '°' : '';
                            return (
                                <div key={i} className="flex items-center gap-2 text-xs">
                                    <span className="text-red-400 font-mono w-4">
                                        {deviationIcon(dev.direction)}
                                    </span>
                                    <span className="text-zinc-300 flex-1 truncate">{label}</span>
                                    <span className="text-zinc-500 font-mono">
                                        {dev.value.toFixed(0)}{unit}
                                    </span>
                                    <span className="text-zinc-600 font-mono text-[10px]">
                                        ({dev.ideal_lo.toFixed(0)}-{dev.ideal_hi.toFixed(0)}{unit})
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Key Joint Angles (when features available) */}
            {bioFeatures && bioFeatures.length >= 16 && (
                <div className="relative bg-zenith-panel/85 border border-zinc-800 p-3 rounded-lg backdrop-blur-sm pointer-events-auto z-20">
                    <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-2 flex items-center">
                        JOINT ANGLES
                        <InfoTooltip text="Real-time joint angles computed from MediaPipe skeletal landmarks. Values in degrees, extracted from the 3D angle at each joint." />
                    </h3>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                        {[
                            { idx: 10, label: 'L Knee' },
                            { idx: 11, label: 'R Knee' },
                            { idx: 6, label: 'L Hip' },
                            { idx: 7, label: 'R Hip' },
                            { idx: 0, label: 'L Shldr' },
                            { idx: 1, label: 'R Shldr' },
                        ].map(({ idx, label }) => {
                            const angle = (bioFeatures[idx] * 180);
                            return (
                                <div key={idx} className="flex justify-between text-xs">
                                    <span className="text-zinc-400">{label}</span>
                                    <span className="text-zinc-200 font-mono">{angle.toFixed(0)}°</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
};
