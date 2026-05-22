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
    
    // Hyper-Lattice Particles
    const nodeCount = 60;
    const nodes: { 
      x: number; y: number; z: number; 
      vx: number; vy: number; vz: number; 
      baseX: number; baseY: number; baseZ: number;
    }[] = [];

    for (let i = 0; i < nodeCount; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos((Math.random() * 2) - 1);
        const radius = 300 + Math.random() * 200;

        nodes.push({
            x: radius * Math.sin(phi) * Math.cos(theta),
            y: radius * Math.sin(phi) * Math.sin(theta),
            z: radius * Math.cos(phi),
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            vz: (Math.random() - 0.5) * 0.5,
            baseX: 0, baseY: 0, baseZ: 0
        });
    }

    let time = 0;

    const animate = () => {
        time += 0.005;
        ctx.fillStyle = "rgba(0, 4, 8, 0.15)"; 
        ctx.fillRect(0, 0, width, height);

        const cx = width / 2;
        const cy = height / 2;

        // Perspective projection
        const focalLength = 600;

        ctx.lineWidth = 0.5;

        // Draw connections (Lattice)
        for (let i = 0; i < nodes.length; i++) {
            const n1 = nodes[i];
            
            // Movement
            n1.x += n1.vx;
            n1.y += n1.vy;
            n1.z += n1.vz;

            // Rotation around Y axis
            const xRot = n1.x * Math.cos(time) - n1.z * Math.sin(time);
            const zRot = n1.x * Math.sin(time) + n1.z * Math.cos(time);
            
            // Rotation around X axis
            const yRot = n1.y * Math.cos(time * 0.5) - zRot * Math.sin(time * 0.5);
            const zFinal = n1.y * Math.sin(time * 0.5) + zRot * Math.cos(time * 0.5);

            const scale = focalLength / (focalLength + zFinal);
            const px1 = xRot * scale + cx;
            const py1 = yRot * scale + cy;

            if (scale > 0) {
                // Connect to nearest neighbors
                for (let j = i + 1; j < nodes.length; j++) {
                    const n2 = nodes[j];
                    const dx = n1.x - n2.x;
                    const dy = n1.y - n2.y;
                    const dz = n1.z - n2.z;
                    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

                    if (dist < 200) {
                        const xRot2 = n2.x * Math.cos(time) - n2.z * Math.sin(time);
                        const zRot2 = n2.x * Math.sin(time) + n2.z * Math.cos(time);
                        const yRot2 = n2.y * Math.cos(time * 0.5) - zRot2 * Math.sin(time * 0.5);
                        const zFinal2 = n2.y * Math.sin(time * 0.5) + zRot2 * Math.cos(time * 0.5);
                        
                        const scale2 = focalLength / (focalLength + zFinal2);
                        const px2 = xRot2 * scale2 + cx;
                        const py2 = yRot2 * scale2 + cy;

                        const opacity = (1 - dist / 200) * 0.3;
                        ctx.strokeStyle = `rgba(0, 255, 255, ${opacity})`;
                        ctx.beginPath();
                        ctx.moveTo(px1, py1);
                        ctx.lineTo(px2, py2);
                        ctx.stroke();
                    }
                }

                // Draw node
                const size = scale * 2;
                ctx.fillStyle = `rgba(0, 200, 255, ${scale * 0.5})`;
                ctx.beginPath();
                ctx.arc(px1, py1, size, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        
        requestAnimationFrame(animate);
    };

    animate();

    const handleResize = () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', handleResize);

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return <canvas ref={canvasRef} id="animated-bg" className="opacity-60 fixed inset-0 pointer-events-none" />;
};

export default AnimatedBackground;