import { useState } from "react";

// Base (Cycles) vs delivered image slider.
export function Compare({ before, after }: { before: string; after: string }) {
  const [pos, setPos] = useState(50);
  return (
    <div className="compare">
      <img src={before} alt="Cycles base render" />
      <div className="compare-top" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
        <img src={after} alt="Delivered render" />
      </div>
      <input
        type="range" min={0} max={100} value={pos} aria-label="Compare base and delivered"
        onChange={(e) => setPos(Number(e.target.value))}
      />
      <div className="compare-labels"><span>Cycles</span><span>Delivered</span></div>
    </div>
  );
}
