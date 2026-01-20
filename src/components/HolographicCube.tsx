
import React, { useRef } from 'react';
import CubeFace from './CubeFace';
import { FaceData } from '../utils/dashboardConfig';

interface HolographicCubeProps {
  currentFace: number;
  rotation: { x: number; y: number };
  faceData: Record<number, FaceData>;
  onMaximize?: (panelId: string) => void;
}

const HolographicCube: React.FC<HolographicCubeProps> = ({ currentFace, rotation, faceData, onMaximize }) => {
  const cubeRef = useRef<HTMLDivElement>(null);

  // The Z-translation here effectively sets the "size" of the cube in 3D space.
  // 50vw ensures it fills the viewport width-wise appropriately for the perspective.
  const translateZ = "translateZ(45vw)"; 
  // On desktop, we might want a fixed pixel depth to prevent it getting too huge
  const desktopTranslateZ = "translateZ(600px)";

  // Helper to construct the transform string for responsive design
  const getTransform = (baseRotate: string) => {
      // In a real app, you might use a media query hook, but CSS min() or clamp() in style is hard for 3D transforms.
      // We'll rely on the CSS class logic or just use a safe large value that works for the container.
      return `${baseRotate} ${desktopTranslateZ}`; 
  };

  return (
    <div className="relative w-full h-full transform-style-preserve-3d transition-transform duration-1000 ease-in-out"
         style={{ 
           transform: `rotateX(${rotation.x}deg) rotateY(${rotation.y}deg)`,
           width: '100%',
           height: '100%'
         }}
         ref={cubeRef}
    >
      {/* Front Face (0) */}
      <CubeFace 
        index={0} 
        active={currentFace === 0} 
        transform={getTransform("")} 
        content={faceData[0]} 
        onMaximize={onMaximize} 
      />
      
      {/* Right Face (1) */}
      <CubeFace 
        index={1} 
        active={currentFace === 1} 
        transform={getTransform("rotateY(90deg)")} 
        content={faceData[1]} 
        onMaximize={onMaximize} 
      />
      
      {/* Back Face (2) */}
      <CubeFace 
        index={2} 
        active={currentFace === 2} 
        transform={getTransform("rotateY(180deg)")} 
        content={faceData[2]} 
        onMaximize={onMaximize} 
      />
      
      {/* Left Face (3) */}
      <CubeFace 
        index={3} 
        active={currentFace === 3} 
        transform={getTransform("rotateY(-90deg)")} 
        content={faceData[3]} 
        onMaximize={onMaximize} 
      />
      
      {/* Top Face (4) */}
      <CubeFace 
        index={4} 
        active={currentFace === 4} 
        transform={getTransform("rotateX(90deg)")} 
        content={faceData[4]} 
        onMaximize={onMaximize} 
      />
      
      {/* Bottom Face (5) */}
      <CubeFace 
        index={5} 
        active={currentFace === 5} 
        transform={getTransform("rotateX(-90deg)")} 
        content={faceData[5]} 
        onMaximize={onMaximize} 
      />
      
      {/* Inner Core: The "Tesseract" singularity */}
      <div className="absolute top-1/2 left-1/2 w-64 h-64 -translate-x-1/2 -translate-y-1/2 pointer-events-none transform-style-preserve-3d animate-spin-slow">
         <div className="absolute inset-0 border border-cyan-500/20 rounded-full animate-pulse"></div>
         <div className="absolute inset-4 border border-purple-500/20 rounded-full animate-spin-reverse-slow"></div>
         <div className="absolute inset-0 bg-cyan-500/5 blur-3xl rounded-full"></div>
      </div>
    </div>
  );
};

export default HolographicCube;
