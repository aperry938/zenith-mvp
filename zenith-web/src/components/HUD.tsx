import React from 'react';

interface ZenithMetrics {
    label: string;
    flow: number;
    velocity: number;
    q: number;
}

interface HUDProps {
    metrics: ZenithMetrics | null;
}

export const HUD: React.FC<HUDProps> = ({ metrics }) => {
    return (
        <div className="hud-panel">
            <div className="hud-item">
                <h3>POSE</h3>
                <div className="hud-value neon-blue">{metrics?.label || "--"}</div>
            </div>
            <div className="hud-item">
                <h3>FLOW</h3>
                <div className="hud-value neon-purple">{metrics?.flow?.toFixed(0) || "--"}</div>
            </div>
            <div className="hud-item">
                <h3>QUALITY</h3>
                <div className="hud-value neon-green">{metrics?.q?.toFixed(0) || "--"}</div>
            </div>
        </div>
    );
};
