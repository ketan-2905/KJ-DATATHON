// import React from 'react';
// import { 
//   Network, 
//   Orbit, 
//   Fingerprint, 
//   Activity, 
//   ArrowRight, 
//   Box, 
//   Zap, 
//   Terminal
// } from 'lucide-react';

// const HyperModernLanding: React.FC = () => {
//   return (
//     <div className="min-h-screen bg-black text-slate-300 font-sans antialiased selection:bg-indigo-500/30">
      
//       {/* Background Glows */}
//       <div className="fixed inset-0 overflow-hidden pointer-events-none">
//         <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] bg-indigo-600/10 blur-[120px] rounded-full" />
//         <div className="absolute top-[20%] -right-[10%] w-[30%] h-[50%] bg-blue-600/5 blur-[120px] rounded-full" />
//       </div>

//       {/* Nav */}
//       <nav className="relative z-50 flex items-center justify-between px-10 py-8 max-w-7xl mx-auto">
//         <div className="flex items-center gap-3">
//           <div className="relative">
//             <div className="absolute inset-0 bg-indigo-500 blur-md opacity-50" />
//             <Orbit className="relative text-white" size={28} />
//           </div>
//           <span className="text-xl font-medium tracking-tighter text-white uppercase">Equilibrium.ai</span>
//         </div>
//         <div className="hidden md:flex gap-8 text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">
//           <a href="#" className="hover:text-white transition-colors">Framework</a>
//           <a href="#" className="hover:text-white transition-colors">The Graph</a>
//           <a href="#" className="hover:text-white transition-colors">Stability</a>
//         </div>
//         <button className="px-5 py-2 rounded-full border border-slate-800 bg-slate-900/50 text-xs font-bold text-white hover:border-slate-700 transition-all">
//           Sign In
//         </button>
//       </nav>

//       {/* Hero Section */}
//       <section className="relative z-10 px-6 pt-32 pb-20 text-center">
//         <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-[10px] font-bold uppercase tracking-[0.2em] text-indigo-400 mb-10">
//           <Zap size={12} /> Systemic Risk Intelligence
//         </div>
        
//         <h1 className="text-6xl md:text-8xl font-bold text-white tracking-tighter mb-8 leading-[0.9]">
//           Infrastructure <br />
//           <span className="text-transparent bg-clip-text bg-gradient-to-b from-white to-slate-500 italic">
//             is a Game.
//           </span>
//         </h1>

//         <p className="max-w-2xl mx-auto text-lg md:text-xl text-slate-400 font-light leading-relaxed mb-12">
//           A network-based modeling engine analyzing strategic interactions among 
//           banks, exchanges, and clearing houses. Predict cascading failures before they happen.
//         </p>

//         <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
//           <button className="group relative px-8 py-4 bg-white text-black rounded-full font-bold transition-all hover:scale-105 active:scale-95">
//             Get Started
//             <div className="absolute inset-0 rounded-full bg-white/20 blur-lg group-hover:blur-xl transition-all opacity-0 group-hover:opacity-100" />
//           </button>
//           <button className="flex items-center gap-2 text-sm font-bold text-slate-400 hover:text-white transition-colors">
//             Read Whitepaper <ArrowRight size={16} />
//           </button>
//         </div>
//       </section>

//       {/* The Bento Framework */}
//       <section className="relative z-10 max-w-7xl mx-auto px-6 py-20">
//         <div className="grid grid-cols-12 gap-4">
          
//           {/* Main Problem Statement Card */}
//           <div className="col-span-12 lg:col-span-8 p-1 rounded-3xl bg-gradient-to-br from-white/10 to-transparent">
//             <div className="h-full bg-black rounded-[calc(1.5rem-1px)] p-10 flex flex-col justify-between overflow-hidden relative">
//               <div className="absolute top-0 right-0 p-8 opacity-20">
//                 <Network size={200} className="text-indigo-500" />
//               </div>
//               <div className="relative z-10">
//                 <h3 className="text-3xl font-bold text-white mb-6 tracking-tight">The Interaction Problem</h3>
//                 <p className="text-slate-400 leading-relaxed max-w-md">
//                   Individual incentives—like credit provision and margin requirements—interact 
//                   through credit exposures. We model these as strategic moves under incomplete 
//                   information to detect systemic bottlenecks.
//                 </p>
//               </div>
//               <div className="mt-20 flex gap-6 text-[10px] font-black uppercase tracking-widest text-indigo-400">
//                 <div className="flex items-center gap-2"><Activity size={14}/> Liquidity Flow</div>
//                 <div className="flex items-center gap-2"><Fingerprint size={14}/> Unique Incentives</div>
//               </div>
//             </div>
//           </div>

//           {/* Impact Stats Card */}
//           <div className="col-span-12 lg:col-span-4 p-[1px] rounded-3xl bg-slate-800">
//             <div className="h-full bg-slate-900/50 rounded-[calc(1.5rem-1px)] p-10 flex flex-col justify-center text-center">
//               <div className="mb-6">
//                 <span className="text-5xl font-bold text-white tracking-tighter">Resilience</span>
//               </div>
//               <p className="text-sm text-slate-500 mb-8 font-mono">
//                 {`$ \Delta \text{Stability} = \sum \text{Node}_i $`}
//               </p>
//               <div className="grid grid-cols-2 gap-2">
//                 <div className="bg-white/5 p-4 rounded-xl border border-white/5 text-xs text-indigo-300">Banks</div>
//                 <div className="bg-white/5 p-4 rounded-xl border border-white/5 text-xs text-indigo-300">Exchanges</div>
//               </div>
//             </div>
//           </div>

//           {/* Business Impact Horizontal */}
//           <div className="col-span-12 p-[1px] rounded-3xl bg-gradient-to-r from-transparent via-slate-800 to-transparent">
//             <div className="bg-black/80 rounded-[calc(1.5rem-1px)] p-12 grid md:grid-cols-4 gap-8">
//               <ImpactItem title="Regulators" desc="Understand micro-decisions and macro-risk." />
//               <ImpactItem title="Risk Mgmt" desc="Detect fragile network structures." />
//               <ImpactItem title="Clearing" desc="Optimize safer settlement mechanisms." />
//               <ImpactItem title="Policy" desc="Draft better regulatory policies." />
//             </div>
//           </div>

//         </div>
//       </section>

//       {/* Terminal View */}
//       <section className="max-w-4xl mx-auto px-6 py-20 text-center">
//         <div className="bg-slate-900/40 rounded-2xl border border-slate-800 p-2 shadow-2xl">
//           <div className="flex gap-1.5 px-4 py-3 border-b border-slate-800">
//             <div className="w-2 h-2 rounded-full bg-red-500/50" />
//             <div className="w-2 h-2 rounded-full bg-yellow-500/50" />
//             <div className="w-2 h-2 rounded-full bg-green-500/50" />
//             <span className="text-[10px] text-slate-500 font-mono ml-4">network_model.tsx</span>
//           </div>
//           <div className="p-8 text-left font-mono text-sm leading-relaxed text-indigo-300/80">
//             <span className="text-indigo-400">const</span> <span className="text-white">analyzeNetwork</span> = (nodes) =&gt; &#123; <br />
//             &nbsp;&nbsp;<span className="text-slate-500">// Propagate localized decisions</span> <br />
//             &nbsp;&nbsp;nodes.forEach(n =&gt; n.playGame(NashEquilibrium)); <br />
//             &nbsp;&nbsp;<span className="text-indigo-400">return</span> detectSystemicRisk(nodes); <br />
//             &#125;
//           </div>
//         </div>
//       </section>

//       {/* Footer */}
//       <footer className="py-20 border-t border-slate-900 text-center">
//         <p className="text-[10px] font-bold uppercase tracking-[0.4em] text-slate-600 italic">
//           High-Fidelity Financial Infrastructure Intelligence • 2026
//         </p>
//       </footer>
//     </div>
//   );
// };

// const ImpactItem = ({ title, desc }: { title: string, desc: string }) => (
//   <div className="space-y-3">
//     <div className="flex items-center gap-2">
//       <div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
//       <h4 className="text-sm font-bold text-white uppercase tracking-wider">{title}</h4>
//     </div>
//     <p className="text-xs text-slate-500 leading-relaxed font-light">{desc}</p>
//   </div>
// );

// export default HyperModernLanding;

"use client"
import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, Sphere, Line, Billboard, Circle } from '@react-three/drei';
import * as THREE from 'three';
import { 
  Network, Orbit, Fingerprint, Activity, ArrowRight, 
  Zap, ShieldCheck, BarChart3, Globe, Database 
} from 'lucide-react';

/** * 3D Network Background Component 
 * Creates a subtle, light-themed moving node network
 */
// const NetworkBackground: React.FC = () => {
//   const count = 30;
//   const lines = useMemo(() => {
//     const temp = [];
//     for (let i = 0; i < count; i++) {
//       temp.push([
//         new THREE.Vector3(Math.random() * 10 - 5, Math.random() * 10 - 5, Math.random() * 10 - 5),
//         new THREE.Vector3(Math.random() * 10 - 5, Math.random() * 10 - 5, Math.random() * 10 - 5)
//       ]);
//     }
//     return temp;
//   }, []);

//   return (
//     <group opacity={0.3}>
//       {lines.map((points, i) => (
//         <Line 
//           key={i} 
//           points={points as [THREE.Vector3, THREE.Vector3]} 
//           color="#6366f1" 
//           lineWidth={0.5} 
//           transparent 
//           opacity={0.1} 
//         />
//       ))}
//       {Array.from({ length: count }).map((_, i) => (
//         <Float key={i} speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
//           <Sphere args={[0.05, 16, 16]} position={[Math.random() * 10 - 5, Math.random() * 10 - 5, Math.random() * 10 - 5]}>
//             <meshStandardMaterial color="#818cf8" emissive="#818cf8" emissiveIntensity={0.5} />
//           </Sphere>
//         </Float>
//       ))}
//     </group>
//   );
// };

// const NetworkBackground: React.FC = () => {
//   const nodeCount = 40; // Number of circular nodes
  
//   // 1. Generate fixed positions for nodes first
//   const nodePositions = useMemo(() => {
//     return Array.from({ length: nodeCount }).map(() => (
//       new THREE.Vector3(
//         (Math.random() - 0.5) * 12, // X range
//         (Math.random() - 0.5) * 10, // Y range
//         (Math.random() - 0.5) * 8   // Z range
//       )
//     ));
//   }, [nodeCount]);

//   // 2. Generate connections between nodes
//   const connections = useMemo(() => {
//     const lines = [];
//     for (let i = 0; i < nodeCount; i++) {
//       // Connect each node to 2-3 of its nearest neighbors for a "proper" network look
//       for (let j = i + 1; j < nodeCount; j++) {
//         const distance = nodePositions[i].distanceTo(nodePositions[j]);
//         if (distance < 4) { // Only connect if nodes are relatively close
//           lines.push([nodePositions[i], nodePositions[j]]);
//         }
//       }
//     }
//     return lines;
//   }, [nodePositions]);

//   return (
//     <group>
//       {/* 3. Render the Connections (Indigo Lines) */}
//       {connections.map((points, i) => (
//         <Line 
//           key={`line-${i}`} 
//           points={points as [THREE.Vector3, THREE.Vector3]} 
//           color="#6366f1" 
//           lineWidth={0.8} 
//           transparent 
//           opacity={0.15} // Kept subtle for light theme
//         />
//       ))}

//       {/* 4. Render the Circular Nodes (Light Indigo Spheres) */}
//       {nodePositions.map((pos, i) => (
//         <Float 
//           key={`node-${i}`} 
//           speed={1.5} 
//           rotationIntensity={0.2} 
//           floatIntensity={0.5} 
//           position={pos.toArray() as [number, number, number]}
//         >
//           <Sphere args={[0.07, 24, 24]}>
//             <meshStandardMaterial 
//               color="#818cf8" 
//               emissive="#818cf8" 
//               emissiveIntensity={0.8} 
//               roughness={0} 
//             />
//           </Sphere>
//         </Float>
//       ))}
//     </group>
//   );
// };

// const NetworkBackground: React.FC = () => {
//   // --- CONFIGURATION VARIABLES ---
//   const config = {
//     nodeCount: 45,           // Number of nodes
//     connectionDist: 4.5,     // Max distance for a connection
//     nodeSize: 0.18,          // Radius of the circle/sphere
//     nodeColor: "#6366f1",    // Primary Indigo
//     lineColor: "#818cf8",    // Lighter Indigo for lines
//     lineOpacity: 0.15,       // Subtle for light theme
//     pulseSpeed: 2,           // Speed of the glowing pulse
//   };

//   // 1. Generate stable node positions
//   const nodePositions = useMemo(() => {
//     return Array.from({ length: config.nodeCount }).map(() => (
//       new THREE.Vector3(
//         (Math.random() - 0.5) * 15,
//         (Math.random() - 0.5) * 10,
//         (Math.random() - 0.5) * 8
//       )
//     ));
//   }, [config.nodeCount]);

//   // 2. Map connections based on distance
//   const connections = useMemo(() => {
//     const lines = [];
//     for (let i = 0; i < config.nodeCount; i++) {
//       for (let j = i + 1; j < config.nodeCount; j++) {
//         const distance = nodePositions[i].distanceTo(nodePositions[j]);
//         if (distance < config.connectionDist) {
//           lines.push([nodePositions[i], nodePositions[j]]);
//         }
//       }
//     }
//     return lines;
//   }, [nodePositions, config.connectionDist]);

//   // 3. Animation Logic (Pulsing and Drifting)
//   const groupRef = useRef<THREE.Group>(null);
  
//   useFrame((state) => {
//     const t = state.clock.getElapsedTime();
//     if (groupRef.current) {
//       // Subtle rotation of the whole network
//       groupRef.current.rotation.y = t * 0.05;
//       groupRef.current.rotation.x = Math.sin(t * 0.1) * 0.1;
//     }
//   });

//   return (
//     <group ref={groupRef}>
//       {/* RENDER CONNECTIONS */}
//       {connections.map((points, i) => (
//         <Line 
//           key={`line-${i}`} 
//           points={points as [THREE.Vector3, THREE.Vector3]} 
//           color={config.lineColor} 
//           lineWidth={0.6} 
//           transparent 
//           opacity={config.lineOpacity} 
//         />
//       ))}

//       {/* RENDER NODES */}
//       {nodePositions.map((pos, i) => (
//         <AnimatedNode 
//           key={`node-${i}`} 
//           position={pos} 
//           size={config.nodeSize} 
//           color={config.nodeColor}
//           pulseSpeed={config.pulseSpeed}
//         />
//       ))}
//     </group>
//   );
// };

const NetworkBackground: React.FC = () => {
  // --- USER CONTROL VARIABLES ---
  const config = {
    nodeCount: 50,           // Total nodes
    connectionDist: 4.0,     // Max distance for a link
    nodeSize: 0.15,          // Size of the circle
    rotationSpeed: 0.2,      // Speed of the full 360-degree spin
    nodeColor: "#4f46e5",    // Primary Indigo
    lineColor: "#c7d2fe",    // Light Indigo/Lavender for lines
    lineOpacity: 0.3,        // Clarity for light theme
  };

  // 1. Generate stable node positions
  const nodePositions = useMemo(() => {
    return Array.from({ length: config.nodeCount }).map(() => (
      new THREE.Vector3(
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 8,
        (Math.random() - 0.5) * 10
      )
    ));
  }, [config.nodeCount]);

  // 2. Map connections based on proximity
  const connections = useMemo(() => {
    const lines = [];
    for (let i = 0; i < config.nodeCount; i++) {
      for (let j = i + 1; j < config.nodeCount; j++) {
        const distance = nodePositions[i].distanceTo(nodePositions[j]);
        if (distance < config.connectionDist) {
          lines.push([nodePositions[i], nodePositions[j]]);
        }
      }
    }
    return lines;
  }, [nodePositions, config.connectionDist]);

  // 3. Animation Logic: Full 360 Rotation
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame((state, delta) => {
    if (groupRef.current) {
      // Continuous 360-degree rotation on Y axis
      groupRef.current.rotation.y += delta * config.rotationSpeed;
      // Optional slight oscillation on X axis for organic feel
      groupRef.current.rotation.x = Math.sin(state.clock.getElapsedTime() * 0.1) * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      {/* RENDER CONNECTIONS */}
      {connections.map((points, i) => (
        <Line 
          key={`line-${i}`} 
          points={points as [THREE.Vector3, THREE.Vector3]} 
          color={config.lineColor} 
          lineWidth={0.8} 
          transparent 
          opacity={config.lineOpacity} 
        />
      ))}

      {/* RENDER NODES AS PERFECT CIRCLES */}
      {nodePositions.map((pos, i) => (
        <Billboard key={`node-${i}`} position={pos}>
          <Circle args={[config.nodeSize, 32]}>
            <meshBasicMaterial 
              color={config.nodeColor} 
              transparent 
              opacity={0.9} 
            />
          </Circle>
          {/* Subtle Outer Glow Layer */}
          <Circle args={[config.nodeSize * 1.5, 32]}>
            <meshBasicMaterial 
              color={config.nodeColor} 
              transparent 
              opacity={0.15} 
            />
          </Circle>
        </Billboard>
      ))}
    </group>
  );
};

/**
 * Individual Node with Pulsing Animation
 */
const AnimatedNode = ({ position, size, color, pulseSpeed }: any) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (meshRef.current) {
      // Create a "breathing" scale effect
      const scale = 1 + Math.sin(t * pulseSpeed + position.x) * 0.2;
      meshRef.current.scale.set(scale, scale, scale);
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.2} floatIntensity={0.5} position={position.toArray()}>
      <Sphere ref={meshRef} args={[size, 32, 32]}>
        <meshStandardMaterial 
          color={color} 
          emissive={color} 
          emissiveIntensity={1.5} // Makes it look like a glowing circle
          toneMapped={false}
        />
      </Sphere>
    </Float>
  );
};

const EquilibriumLanding: React.FC = () => {
  return (
    <div className="min-h-screen bg-white text-slate-900 font-sans antialiased selection:bg-indigo-100 overflow-x-hidden">
      
      {/* Navigation */}
      <nav className="sticky top-0 z-50 flex items-center justify-between px-10 py-5 bg-white/80 backdrop-blur-md border-b border-slate-100">
        <div className="flex items-center gap-2.5">
          <div className="bg-slate-900 p-1.5 rounded-lg shadow-lg">
            <Orbit className="text-white" size={22} />
          </div>
          <span className="text-lg font-bold tracking-tighter uppercase text-slate-900">Equilibrium.ai</span>
        </div>
        <div className="hidden md:flex gap-10 text-[11px] font-bold uppercase tracking-[0.2em] text-slate-400">
          <a href="#model" className="hover:text-slate-900 transition-colors">The Model</a>
          <a href="#logic" className="hover:text-slate-900 transition-colors">Equilibrium Logic</a>
          <a href="#impact" className="hover:text-slate-900 transition-colors">Impact</a>
        </div>
        <button className="px-6 py-2.5 rounded-full bg-slate-900 text-white text-xs font-bold hover:shadow-lg transition-all active:scale-95">
          Platform Demo
        </button>
      </nav>

      {/* Heroic Section with 3D Background */}
      <section className="relative z-10 h-[85vh] flex flex-col items-center justify-center text-center px-6">
        <div className="absolute inset-0 z-0 opacity-40">
          <Canvas camera={{ position: [0, 0, 8] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <NetworkBackground />
          </Canvas>
        </div>

        <div className="relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-[10px] font-black uppercase tracking-[0.2em] text-indigo-600 mb-8 shadow-sm">
            <Zap size={12} fill="currentColor" /> Systemic Risk Intelligence
          </div>
          
          <h1 className="text-7xl md:text-9xl font-bold text-slate-900 tracking-tighter mb-8 leading-[0.85]">
            Infrastructure <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-b from-slate-900 to-slate-500 italic">
              is a Game.
            </span>
          </h1>

          <p className="max-w-2xl mx-auto text-xl text-slate-500 font-medium leading-relaxed mb-12">
            A network-based modeling engine analyzing strategic interactions among 
            global institutions. Predict cascading failures with computational precision.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
            <button className="px-10 py-5 bg-slate-900 text-white rounded-full font-bold text-lg hover:bg-black transition-all shadow-2xl shadow-slate-400/20 active:scale-95">
              Launch Framework
            </button>
            <button className="flex items-center gap-2 text-sm font-bold text-slate-500 hover:text-slate-900 transition-colors">
              Technical Documentation <ArrowRight size={18} />
            </button>
          </div>
        </div>
      </section>

 <section className="relative z-10 max-w-7xl mx-auto px-6 py-20">
        <div className="grid grid-cols-12 gap-6">
          
          {/* Bento Card: The Logic Map */}
          <div className="col-span-12 lg:col-span-7 bg-white p-10 rounded-[2.5rem] border border-slate-200 shadow-sm flex flex-col justify-between group hover:border-indigo-200 transition-colors">
            <div>
              <div className="w-12 h-12 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600 mb-8">
                <Network size={28} />
              </div>
              <h3 className="text-3xl font-bold text-slate-900 mb-6 tracking-tight">The Network Interaction Model</h3>
              <p className="text-slate-500 leading-relaxed max-w-lg mb-8">
                We capture how local decisions—credit provision, margin requirements, 
                and trade routing—interact through network connections like credit 
                exposures and settlement obligations.
              </p>
            </div>
            <div className="grid grid-cols-3 gap-4 border-t border-slate-100 pt-8">
                <div className="text-center">
                    <p className="text-xs font-bold uppercase text-slate-400 mb-1">Incentives</p>
                    <p className="text-lg font-bold text-slate-900 italic">Strategic</p>
                </div>
                <div className="text-center border-x border-slate-100">
                    <p className="text-xs font-bold uppercase text-slate-400 mb-1">Information</p>
                    <p className="text-lg font-bold text-slate-900 italic">Incomplete</p>
                </div>
                <div className="text-center">
                    <p className="text-xs font-bold uppercase text-slate-400 mb-1">Outcome</p>
                    <p className="text-lg font-bold text-slate-900 italic">Resilient</p>
                </div>
            </div>
          </div>

          {/* Side Card: Systemic Impact */}
          <div className="col-span-12 lg:col-span-5 bg-slate-50 p-10 rounded-[2.5rem] border border-slate-200 flex flex-col justify-center overflow-hidden relative">
            <div className="absolute top-[-20%] right-[-20%] opacity-5">
                <Globe size={300} />
            </div>
            <div className="relative z-10">
                <h4 className="text-sm font-black uppercase tracking-widest text-indigo-600 mb-4">Macro Outcomes</h4>
                <div className="space-y-6">
                    <ImpactMetric label="Liquidity Flow" value="Optimized" color="bg-emerald-500" />
                    <ImpactMetric label="Congestion Risk" value="Minimal" color="bg-indigo-500" />
                    <ImpactMetric label="Financial Stability" value="Validated" color="bg-cyan-500" />
                </div>
                <div className="mt-10 p-5 bg-white rounded-2xl border border-slate-200 font-mono text-[11px] text-slate-400">
                    
                    <br />
                   {" $ \text{Stability} = f(\text{micro\_decisions}) $"}
                </div>
            </div>
          </div>

          {/* Business Impact Row */}
          <div className="col-span-12 grid md:grid-cols-4 gap-6 mt-6">
             <ImpactBox 
                icon={<ShieldCheck className="text-indigo-600" />} 
                title="Regulators" 
                desc="Understand how micro-level decisions create macro-level risks." 
             />
             <ImpactBox 
                icon={<Activity className="text-emerald-600" />} 
                title="Institutions" 
                desc="Detect fragile structures and bottlenecks in settlement." 
             />
             <ImpactBox 
                icon={<BarChart3 className="text-cyan-600" />} 
                title="Management" 
                desc="Improve risk management against economic shocks." 
             />
             <ImpactBox 
                icon={<Database className="text-slate-600" />} 
                title="Clearing" 
                desc="Support safer mechanisms and aligned incentives." 
             />
          </div>
        </div>
      </section>

      {/* Proper Footer Section */}
      <footer className="bg-slate-50 border-t border-slate-200 pt-20 pb-12">
        <div className="max-w-7xl mx-auto px-10">
          <div className="grid md:grid-cols-4 gap-12 mb-16">
            <div className="col-span-1 md:col-span-2">
                <div className="flex items-center gap-2 mb-6">
                    <Orbit className="text-slate-900" size={24} />
                    <span className="text-xl font-bold tracking-tighter uppercase">Equilibrium</span>
                </div>
                <p className="max-w-xs text-sm text-slate-500 leading-relaxed italic font-serif">
                    "Analyzing individual incentives to safeguard global financial infrastructure through network-based game theory."
                </p>
            </div>
            <div>
                <h5 className="text-xs font-black uppercase tracking-widest text-slate-900 mb-6">Framework</h5>
                <ul className="text-sm text-slate-500 space-y-4 font-medium">
                    <li className="hover:text-indigo-600 cursor-pointer">Nash Equilibrium</li>
                    <li className="hover:text-indigo-600 cursor-pointer">Network Propagation</li>
                    <li className="hover:text-indigo-600 cursor-pointer">Settlement Logic</li>
                </ul>
            </div>
            <div>
                <h5 className="text-xs font-black uppercase tracking-widest text-slate-900 mb-6">Legal</h5>
                <ul className="text-sm text-slate-500 space-y-4 font-medium">
                    <li className="hover:text-indigo-600 cursor-pointer">Privacy Policy</li>
                    <li className="hover:text-indigo-600 cursor-pointer">Terms of Service</li>
                    <li className="hover:text-indigo-600 cursor-pointer">Regulatory Compliance</li>
                </ul>
            </div>
          </div>
          <div className="pt-8 border-t border-slate-200 flex flex-col md:flex-row justify-between items-center gap-6">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.4em]">
              © 2026 Financial Infrastructure Modeling Inc.
            </p>
            <div className="flex gap-6 text-slate-400">
                <Fingerprint size={20} className="hover:text-indigo-600 cursor-pointer transition-colors" />
                <Globe size={20} className="hover:text-indigo-600 cursor-pointer transition-colors" />
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

const ImpactMetric = ({ label, value, color }: { label: string, value: string, color: string }) => (
    <div className="flex justify-between items-center group">
        <div className="flex items-center gap-3">
            <div className={`w-1.5 h-1.5 rounded-full ${color}`} />
            <span className="text-sm font-bold text-slate-600">{label}</span>
        </div>
        <span className="text-xs font-black uppercase text-slate-400 group-hover:text-slate-900 transition-colors">{value}</span>
    </div>
);

const ImpactBox = ({ icon, title, desc }: { icon: React.ReactNode, title: string, desc: string }) => (
  <div className="p-8 bg-white border border-slate-200 rounded-[2rem] hover:shadow-lg hover:shadow-slate-200/50 transition-all">
    <div className="mb-4">{icon}</div>
    <h4 className="text-sm font-black uppercase tracking-widest text-slate-900 mb-3">{title}</h4>
    <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
  </div>
);

export default EquilibriumLanding;

