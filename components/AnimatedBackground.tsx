import React, { useEffect, useRef } from 'react';

const AnimatedBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);
    const nodeCount = 60;
    const nodes: any[] = [];

    for (let i = 0; i < nodeCount; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos((Math.random() * 2) - 1);
        const radius = 300 + Math.random() * 200;
        nodes.push({
            x: radius * Math.sin(phi) * Math.cos(theta),
            y: radius * Math.sin(phi) * Math.sin(theta),
            z: radius * Math.cos(phi),
            vx: (Math.random() - 0.5) * 0.5, vy: (Math.random() - 0.5) * 0.5, vz: (Math.random() - 0.5) * 0.5
        });
    }

    let time = 0;
    const animate = () => {
        time += 0.005;
        ctx.fillStyle = "rgba(0, 4, 8, 0.15)"; ctx.fillRect(0, 0, width, height);
        const cx = width / 2; const cy = height / 2; const focalLength = 600;

        for (let i = 0; i < nodes.length; i++) {
            const n1 = nodes[i];
            n1.x += n1.vx; n1.y += n1.vy; n1.z += n1.vz;
            const xRot = n1.x * Math.cos(time) - n1.z * Math.sin(time);
            const zRot = n1.x * Math.sin(time) + n1.z * Math.cos(time);
            const scale = focalLength / (focalLength + zRot);
            const px1 = xRot * scale + cx; const py1 = n1.y * scale + cy;

            if (scale > 0) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const n2 = nodes[j];
                    const dist = Math.sqrt((n1.x-n2.x)**2 + (n1.y-n2.y)**2 + (n1.z-n2.z)**2);
                    if (dist < 200) {
                        const scale2 = focalLength / (focalLength + n2.z);
                        ctx.strokeStyle = `rgba(0, 255, 255, ${(1 - dist / 200) * 0.3})`;
                        ctx.beginPath(); ctx.moveTo(px1, py1); ctx.lineTo(n2.x * scale2 + cx, n2.y * scale2 + cy); ctx.stroke();
                    }
                }
                ctx.fillStyle = `rgba(0, 200, 255, ${scale * 0.5})`;
                ctx.beginPath(); ctx.arc(px1, py1, scale * 2, 0, Math.PI * 2); ctx.fill();
            }
        }
        requestAnimationFrame(animate);
    };
    animate();
    const handleResize = () => { width = canvas.width = window.innerWidth; height = canvas.height = window.innerHeight; };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return <canvas ref={canvasRef} id="animated-bg" className="opacity-60 fixed inset-0 pointer-events-none z-0" />;
};
export default AnimatedBackground;