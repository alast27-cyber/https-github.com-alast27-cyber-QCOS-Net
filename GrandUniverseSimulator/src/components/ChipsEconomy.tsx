
import React, { useState, useEffect } from 'react';
import GlassPanel from './GlassPanel';
import { 
    BanknotesIcon, WalletIcon, RefreshCwIcon, ChartBarIcon, 
    ShieldCheckIcon, CheckCircle2Icon, ActivityIcon, LockIcon,
    UsersIcon, CpuChipIcon, CubeTransparentIcon, ZapIcon, ServerCogIcon, ScaleIcon, FileIcon,
    AlertTriangleIcon, DocumentArrowUpIcon, StarIcon, TrophyIcon, GiftIcon, SparklesIcon,
    BriefcaseIcon, RocketLaunchIcon, CodeBracketIcon, AtomIcon, GalaxyIcon, ArrowRightIcon
} from './Icons';
import { useToast } from '../context/ToastContext';

type EconomyTab = 'currency' | 'wallet' | 'exchange' | 'cychips' | 'legal' | 'rewards';
type EncryptionState = 'superposition' | 'threat' | 'collapsed' | 'teleported';

const CyChipsEncryptionModule: React.FC = () => {
    const [encState, setEncState] = useState<EncryptionState>('superposition');

    const handleSimulateAttack = () => {
        if (encState !== 'superposition') return;
        
        setEncState('threat');
        
        setTimeout(() => {
            setEncState('collapsed');
        }, 1500);

        setTimeout(() => {
            setEncState('teleported');
        }, 2500);

        setTimeout(() => {
            setEncState('superposition');
        }, 6000);
    };

    return (
        <div className="bg-black/30 border border-cyan-800/50 rounded-xl p-4 mt-6 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent opacity-50"></div>
            
            <div className="flex justify-between items-start mb-6">
                <div>
                    <h3 className="text-lg font-bold text-white flex items-center gap-2">
                        <CubeTransparentIcon className="w-5 h-5 text-purple-400" />
                        CyChips ChainLink Encryption
                    </h3>
                    <p className="text-xs text-cyan-400 mt-1 max-w-lg">
                        Mining generates an encrypted certificate (Coin) composed of 4 fractal quarters. 
                        The <strong>ChainLink</strong> protocol uses quantum entanglement to bind these quarters.
                    </p>
                </div>
                <div className={`px-3 py-1 rounded border text-xs font-bold uppercase tracking-wider transition-colors duration-500
                    ${encState === 'superposition' ? 'bg-green-900/30 border-green-500 text-green-400' : 
                      encState === 'threat' ? 'bg-red-900/50 border-red-500 text-red-200 animate-pulse' : 
                      'bg-purple-900/30 border-purple-500 text-purple-300'}`}>
                    Status: {encState === 'superposition' ? 'Coherent' : encState === 'threat' ? 'THREAT DETECTED' : 'Teleporting...'}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                <div className="relative h-64 bg-black/40 rounded-lg border border-cyan-900/30 flex items-center justify-center overflow-hidden group">
                    <div className="absolute inset-0 holographic-grid opacity-20"></div>
                    <div className={`relative w-40 h-40 transition-all duration-500 ${encState === 'teleported' ? 'opacity-0 scale-0' : 'opacity-100 scale-100'}`}>
                        <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 z-20 transition-all duration-300
                            ${encState === 'threat' ? 'scale-125 drop-shadow-[0_0_15px_rgba(239,68,68,0.8)]' : 'scale-100 drop-shadow-[0_0_10px_rgba(168,85,247,0.6)]'}
                        `}>
                            <CubeTransparentIcon className={`w-full h-full ${encState === 'threat' ? 'text-red-500 animate-spin' : 'text-purple-400 animate-pulse-slow'}`} />
                        </div>
                        <div className={`absolute top-0 left-0 w-1/2 h-1/2 bg-cyan-900/80 border-t-2 border-l-2 border-cyan-400 rounded-tl-full flex items-center justify-center transition-all duration-700 ease-out z-10
                            ${encState === 'collapsed' ? '-translate-x-12 -translate-y-12 rotate-[-45deg] opacity-0' : 
                              encState === 'threat' ? '-translate-x-1 -translate-y-1' : 'translate-x-0 translate-y-0'}
                        `}>
                            <span className="text-[10px] text-cyan-200 font-mono -mt-2 -ml-2">Q1</span>
                        </div>
                        <div className={`absolute top-0 right-0 w-1/2 h-1/2 bg-cyan-900/80 border-t-2 border-r-2 border-cyan-400 rounded-tr-full flex items-center justify-center transition-all duration-700 ease-out z-10
                            ${encState === 'collapsed' ? 'translate-x-12 -translate-y-12 rotate-[45deg] opacity-0' : 
                              encState === 'threat' ? 'translate-x-1 -translate-y-1' : 'translate-x-0 translate-y-0'}
                        `}>
                            <span className="text-[10px] text-cyan-200 font-mono -mt-2 -mr-2">Q2</span>
                        </div>
                        <div className={`absolute bottom-0 left-0 w-1/2 h-1/2 bg-cyan-900/80 border-b-2 border-l-2 border-cyan-400 rounded-bl-full flex items-center justify-center transition-all duration-700 ease-out z-10
                            ${encState === 'collapsed' ? '-translate-x-12 translate-y-12 rotate-[45deg] opacity-0' : 
                              encState === 'threat' ? '-translate-x-1 translate-y-1' : 'translate-x-0 translate-y-0'}
                        `}>
                            <span className="text-[10px] text-cyan-200 font-mono -mb-2 -ml-2">Q3</span>
                        </div>
                        <div className={`absolute bottom-0 right-0 w-1/2 h-1/2 bg-cyan-900/80 border-b-2 border-r-2 border-cyan-400 rounded-br-full flex items-center justify-center transition-all duration-700 ease-out z-10
                            ${encState === 'collapsed' ? 'translate-x-12 translate-y-12 rotate-[-45deg] opacity-0' : 
                              encState === 'threat' ? 'translate-x-1 -translate-y-1' : 'translate-x-0 translate-y-0'}
                        `}>
                            <span className="text-[10px] text-cyan-200 font-mono -mb-2 -mr-2">Q4</span>
                        </div>
                    </div>
                    {encState === 'threat' && <div className="absolute inset-0 bg-red-500/10 z-0 animate-pulse"></div>}
                    {encState === 'collapsed' && (
                        <div className="absolute inset-0 flex items-center justify-center">
                            <div className="w-full h-1 bg-cyan-400 blur-md animate-ping"></div>
                        </div>
                    )}
                </div>

                <div className="flex flex-col gap-4">
                    <div className={`p-4 rounded-lg border-2 transition-all duration-500 flex items-center gap-4 relative overflow-hidden
                        ${encState === 'teleported' ? 'bg-green-900/30 border-green-500 shadow-[0_0_20px_theme(colors.green.600/40%)]' : 'bg-black/40 border-cyan-900/30'}`}>
                        <div className={`p-3 rounded-full ${encState === 'teleported' ? 'bg-green-500 text-black' : 'bg-gray-800 text-gray-500'}`}>
                            <LockIcon className="w-6 h-6" />
                        </div>
                        <div>
                            <h4 className={`text-sm font-bold ${encState === 'teleported' ? 'text-green-300' : 'text-gray-400'}`}>
                                {encState === 'teleported' ? 'Secure Vault: Assets Teleported' : 'Secure Vault: Standby'}
                            </h4>
                            <p className="text-[10px] text-cyan-600 mt-1">Destination: Cold Storage (Quantum-Resistant)</p>
                        </div>
                        {encState === 'teleported' && (
                            <div className="absolute right-4 flex gap-1">
                                <div className="w-3 h-3 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0s'}}></div>
                                <div className="w-3 h-3 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                                <div className="w-3 h-3 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                                <div className="w-3 h-3 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.3s'}}></div>
                            </div>
                        )}
                    </div>
                    <div className="bg-purple-900/20 p-3 rounded border border-purple-700/30 text-xs text-purple-200">
                        <strong className="text-white block mb-1">Physics-Based Security</strong>
                        <p>The ChainLink acts as the observer. If a threat attempts to measure (hack) the coin, the superposition collapses immediately. The 4 quarters unbind and teleport to the vault before data extraction is possible.</p>
                    </div>
                    <button 
                        onClick={handleSimulateAttack}
                        disabled={encState !== 'superposition'}
                        className="holographic-button py-2 bg-red-900/20 border-red-500/50 hover:bg-red-900/40 text-red-200 text-xs font-bold flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <ZapIcon className="w-4 h-4" /> Simulate Quantum Threat
                    </button>
                </div>
            </div>
        </div>
    );
};

const CyChipsRewardPanel: React.FC = () => {
    const { addToast } = useToast();
    const [balance, setBalance] = useState(100);
    const [inviteCount, setInviteCount] = useState(124);
    const [currentValue, setCurrentValue] = useState(0.024); // Start low

    const missions = [
        { id: 'm1', title: 'App Architect', desc: 'Deploy an AGI-grade application node.', reward: '50.0 CYC', icon: RocketLaunchIcon, progress: 0, color: 'text-pink-400' },
        { id: 'm2', title: 'Linguistic Pioneer', desc: 'Submit a Bayq-Lang functional snippet.', reward: '20.0 CYC', icon: CodeBracketIcon, progress: 100, color: 'text-purple-400' },
        { id: 'm3', title: 'Security Sentinel', desc: 'Execute a full system diagnostic cycle.', reward: '10.0 CYC', icon: ShieldCheckIcon, progress: 45, color: 'text-green-400' },
        { id: 'm4', title: 'Cosmological Insight', desc: 'Forecast 3 multiverse timelines.', reward: '15.0 CYC', icon: GalaxyIcon, progress: 12, color: 'text-blue-400' },
    ];

    const handleClaim = (amount: string) => {
        const val = parseFloat(amount.split(' ')[0]);
        setBalance(prev => prev + val);
        addToast(`Claimed ${amount}! Proof of Work verified.`, 'success');
    };

    return (
        <div className="space-y-6 animate-fade-in text-cyan-200 h-full overflow-y-auto custom-scrollbar">
            {/* Mission Statement Hero */}
            <div className="bg-gradient-to-br from-cyan-900/50 to-purple-900/30 p-6 rounded-2xl border border-cyan-500/40 relative overflow-hidden group">
                <div className="absolute inset-0 holographic-grid opacity-10"></div>
                <div className="relative z-10 flex flex-col md:flex-row items-center gap-8">
                    <div className="flex-shrink-0 relative">
                        <div className="absolute inset-0 bg-yellow-400/20 blur-2xl rounded-full animate-pulse"></div>
                        <div className="w-24 h-24 rounded-full bg-black/60 border-2 border-yellow-500/50 flex items-center justify-center relative shadow-[0_0_30px_rgba(234,179,8,0.2)]">
                            <TrophyIcon className="w-12 h-12 text-yellow-400 animate-bounce" />
                        </div>
                    </div>
                    <div className="flex-grow text-center md:text-left">
                        <h2 className="text-2xl font-black text-white tracking-[0.2em] mb-2 uppercase italic flex items-center gap-3 justify-center md:justify-start">
                            CyChips Reward Matrix
                            <span className="bg-green-600 px-2 py-0.5 rounded text-[8px] font-mono not-italic tracking-normal">GENESIS_ENABLED</span>
                        </h2>
                        <p className="text-sm text-cyan-100 max-w-2xl leading-relaxed">
                            The current digital infrastructure is no match for AGI. By building <strong>Chips://</strong>, we are securing an abundant future. 
                            Participate, create, play, and dream—every unit of actual work adds immutable value to the ecosystem.
                        </p>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-1 space-y-4">
                    {/* User Wallet Card */}
                    <div className="bg-black/40 border border-cyan-800 p-5 rounded-2xl flex flex-col items-center text-center relative group overflow-hidden">
                        <div className="absolute inset-0 bg-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                        <p className="text-[10px] font-black text-cyan-500 uppercase tracking-widest mb-4">Node Reward Balance</p>
                        <div className="flex items-baseline gap-2 mb-1">
                            <span className="text-5xl font-mono font-black text-white">{balance.toFixed(1)}</span>
                            <span className="text-xs font-bold text-cyan-400">CyC</span>
                        </div>
                        <p className="text-xs text-gray-500 font-mono italic">≈ ${ (balance * currentValue).toFixed(2) } USD</p>
                        
                        <div className="mt-6 w-full pt-4 border-t border-cyan-900/50 flex justify-between items-center">
                            <div className="text-left">
                                <p className="text-[8px] text-gray-500 uppercase">Invites Used</p>
                                <p className="text-sm font-bold text-white">{inviteCount}/500</p>
                            </div>
                            <div className="text-right">
                                <p className="text-[8px] text-gray-500 uppercase">Pool Share</p>
                                <p className="text-sm font-bold text-cyan-400">0.24%</p>
                            </div>
                        </div>
                        <button className="mt-4 w-full py-2 bg-yellow-600/20 border border-yellow-500/50 rounded text-[10px] font-black text-yellow-200 uppercase hover:bg-yellow-600/30 transition-all flex items-center justify-center gap-2">
                            <UsersIcon className="w-3 h-3" /> Invite Developer
                        </button>
                    </div>

                    {/* Convergence Gauge */}
                    <div className="bg-black/40 border border-cyan-900 p-4 rounded-2xl">
                        <div className="flex justify-between items-center mb-3">
                            <p className="text-[10px] font-black text-cyan-500 uppercase">Target Convergence</p>
                            <span className="text-[10px] font-mono text-green-400">{(currentValue * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden border border-white/5 relative">
                            <div className="absolute inset-0 bg-cyan-500/10 animate-pulse"></div>
                            <div className="h-full bg-gradient-to-r from-cyan-600 to-green-500 shadow-[0_0_15px_rgba(34,197,94,0.3)] transition-all duration-1000" style={{ width: `${currentValue * 100}%` }}></div>
                        </div>
                        <div className="flex justify-between mt-2 text-[8px] font-mono text-gray-600">
                            <span>$0.00</span>
                            <span className="text-white font-bold">PATH TO $1.00 USD</span>
                            <span>$1.00</span>
                        </div>
                        <p className="text-[9px] text-cyan-700 mt-4 text-center leading-relaxed">All income from Chips Store and SoftwareQ remains in the ecosystem as backing value for CyChips.</p>
                    </div>
                </div>

                {/* Mission Board */}
                <div className="md:col-span-2 flex flex-col gap-4">
                    <h3 className="text-xs font-black text-white uppercase tracking-widest flex items-center justify-between px-2">
                        <div className="flex items-center gap-2">
                            <ActivityIcon className="w-4 h-4 text-cyan-400" />
                            Work-Based Reward Vectors
                        </div>
                        <span className="text-[9px] font-mono text-cyan-700">Proof of Participation (PoP)</span>
                    </h3>

                    <div className="grid gap-3">
                        {missions.map(mission => (
                            <div key={mission.id} className="bg-black/40 border border-cyan-900/50 rounded-xl p-4 flex items-center gap-4 group hover:bg-cyan-900/10 hover:border-cyan-500 transition-all duration-300">
                                <div className={`p-3 rounded-lg bg-black/40 border border-gray-800 group-hover:border-cyan-500/30 transition-colors ${mission.color}`}>
                                    <mission.icon className="w-6 h-6" />
                                </div>
                                <div className="flex-grow min-w-0">
                                    <div className="flex justify-between items-center mb-1">
                                        <h5 className="font-bold text-sm text-white group-hover:text-cyan-200 transition-colors">{mission.title}</h5>
                                        <div className="flex items-center gap-2">
                                            <span className="text-[10px] font-mono font-black text-green-400 bg-green-950/40 px-2 rounded border border-green-800">+{mission.reward}</span>
                                        </div>
                                    </div>
                                    <p className="text-[10px] text-gray-500 truncate mb-2">{mission.desc}</p>
                                    <div className="flex items-center gap-3">
                                        <div className="flex-grow h-1 bg-gray-800 rounded-full overflow-hidden">
                                            <div className={`h-full transition-all duration-1000 ${mission.progress === 100 ? 'bg-green-500' : 'bg-cyan-500 shadow-[0_0_8px_cyan]'}`} style={{width: `${mission.progress}%` }}></div>
                                        </div>
                                        <span className="text-[9px] font-mono text-gray-600">{mission.progress}%</span>
                                    </div>
                                </div>
                                {mission.progress === 100 ? (
                                    <button 
                                        onClick={() => handleClaim(mission.reward)}
                                        className="flex-shrink-0 p-2.5 rounded-lg bg-green-600/30 border border-green-500 text-green-400 hover:bg-green-600 hover:text-white transition-all shadow-[0_0_15px_rgba(34,197,94,0.2)] active:scale-90"
                                    >
                                        <GiftIcon className="w-5 h-5 animate-pulse" />
                                    </button>
                                ) : (
                                    <button className="flex-shrink-0 p-2 rounded-lg bg-black/40 border border-cyan-900 text-cyan-700 opacity-30 cursor-not-allowed">
                                        <ArrowRightIcon className="w-5 h-5" />
                                    </button>
                                )}
                            </div>
                        ))}
                    </div>

                    <div className="mt-auto bg-purple-900/10 border border-purple-500/30 p-4 rounded-xl flex items-center gap-4">
                        <div className="p-3 bg-purple-500/20 rounded-full text-purple-400">
                            <SparklesIcon className="w-6 h-6" />
                        </div>
                        <div>
                            <p className="text-xs font-bold text-white mb-1">Genesis Invite Bonus</p>
                            <p className="text-[10px] text-purple-200">First 500 early invitees receive 100.0 CyC instantly to fuel the network. (Status: CLAIMED)</p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="text-center pt-8 border-t border-cyan-900/30">
                <p className="text-[9px] text-cyan-800 uppercase tracking-[0.5em] font-black">Building abundance through sovereign intelligence</p>
            </div>
        </div>
    );
};

const ChipsEconomy: React.FC = () => {
    const [activeTab, setActiveTab] = useState<EconomyTab>('rewards');
    const [submissionState, setSubmissionState] = useState<'idle' | 'submitting' | 'submitted'>('idle');

    // Mock Data for Currency
    const currencyData = {
        name: 'Q-Credits',
        symbol: 'QCR',
        totalSupply: '100,000,000',
        circulating: '42,500,000',
        mintingRate: '12.5 QCR / Block',
        stabilityIndex: '99.8%'
    };

    // Mock Data for CyChips
    const cyChipsData = {
        price: '1.24', 
        marketCap: '$52.7M',
        volume: '1.2M CYC',
        change: '+5.4%'
    };

    // Mock Data for Wallet
    const walletData = {
        gatewayStatus: 'Online (Quantum-Secured)',
        throughput: '14,200 TPS',
        escrowVolume: '1.2M QCR',
        fraudChecks: '0 Detected'
    };

    // Mock Data for Exchange
    const exchangeData = {
        pair: 'CYC / Compute',
        rate: '1 CYC = 100 QPU-Hours',
        liquidity: 'High ($4.5M)',
        volume24h: '850k CYC'
    };

    const handleSubmission = () => {
        setSubmissionState('submitting');
        setTimeout(() => {
            setSubmissionState('submitted');
        }, 2500);
    };

    const renderLegalities = () => (
        <div className="space-y-6 animate-fade-in h-full overflow-y-auto custom-scrollbar pr-2">
            <div className="bg-gradient-to-br from-slate-900 to-black p-4 rounded-xl border border-cyan-700/50 shadow-lg flex flex-col md:flex-row justify-between items-center gap-4">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-cyan-900/30 rounded-full border border-cyan-500/30">
                        <ScaleIcon className="w-8 h-8 text-cyan-400" />
                    </div>
                    <div>
                        <h3 className="text-lg font-bold text-white tracking-wide">Legal Framework & Governance</h3>
                        <p className="text-xs text-cyan-500">Jurisdiction: Decentralized Quantum Mesh (DQM)</p>
                    </div>
                </div>
                <div className="flex items-center gap-3 bg-black/40 px-4 py-2 rounded-lg border border-cyan-900">
                    <div className={`w-3 h-3 rounded-full ${submissionState === 'submitted' ? 'bg-green-500' : 'bg-yellow-500 animate-pulse'}`}></div>
                    <span className="text-sm font-mono text-cyan-200 uppercase">{submissionState === 'submitted' ? 'Ratified & Active' : 'Drafting Phase'}</span>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-black/20 p-4 rounded-lg border border-cyan-800/50 flex flex-col">
                    <h4 className="text-sm font-bold text-cyan-300 mb-3 uppercase tracking-wider flex items-center gap-2">
                        <FileIcon className="w-4 h-4"/> Core Documents
                    </h4>
                    <div className="space-y-2 flex-grow">
                        {['Constitution of CyChips', 'Tokenomics Charter v2.1', 'Privacy & Data Sovereignty Act'].map((doc, i) => (
                            <div key={i} className="flex justify-between items-center p-2 bg-cyan-950/20 rounded border border-cyan-900/30 hover:bg-cyan-900/40 transition-colors cursor-pointer group">
                                <span className="text-xs text-gray-300 group-hover:text-white transition-colors">{doc}</span>
                                <span className="text-[10px] text-green-400 bg-green-900/20 px-2 py-0.5 rounded border border-green-800/50">VERIFIED</span>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-black/20 p-4 rounded-lg border border-cyan-800/50 flex flex-col">
                    <h4 className="text-sm font-bold text-cyan-300 mb-3 uppercase tracking-wider flex items-center gap-2">
                        <ShieldCheckIcon className="w-4 h-4"/> Smart Contract Audits
                    </h4>
                    <div className="space-y-3">
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-gray-400">Code Integrity</span>
                                <span className="text-green-400 font-bold">100%</span>
                            </div>
                            <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                                <div className="bg-green-500 h-full w-full"></div>
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-gray-400">Quantum Resistance</span>
                                <span className="text-green-400 font-bold">98.5%</span>
                            </div>
                            <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                                <div className="bg-green-500 h-full w-[98.5%]"></div>
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-gray-400">Governance Compliance</span>
                                <span className="text-yellow-400 font-bold">Pending Review</span>
                            </div>
                            <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                                <div className="bg-yellow-500 h-full w-[60%] animate-pulse"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-gradient-to-r from-purple-900/20 to-cyan-900/20 p-6 rounded-lg border border-cyan-700/50 text-center">
                <h4 className="text-lg font-bold text-white mb-2">Network Ratification</h4>
                <p className="text-sm text-cyan-300 mb-6 max-w-2xl mx-auto">
                    By submitting the legal framework, you initiate the decentralized voting protocol. All active nodes will verify the integrity of the documents and the compliance of the smart contracts against the QCOS Constitution.
                </p>
                <button 
                    onClick={handleSubmission}
                    disabled={submissionState !== 'idle'}
                    className={`holographic-button px-8 py-3 rounded text-sm font-bold flex items-center justify-center gap-2 mx-auto transition-all ${submissionState === 'submitted' ? 'bg-green-600/30 border-green-500 text-green-300 cursor-default' : 'bg-cyan-600/30 border-cyan-500 text-cyan-200 hover:bg-cyan-600/50'}`}
                >
                    {submissionState === 'idle' && <><DocumentArrowUpIcon className="w-5 h-5"/> Submit Framework</>}
                    {submissionState === 'submitting' && <><RefreshCwIcon className="w-5 h-5 animate-spin"/> Verifying Chain...</>}
                    {submissionState === 'submitted' && <><CheckCircle2Icon className="w-5 h-5"/> Submission Complete</>}
                </button>
            </div>
        </div>
    );

    const renderCyChips = () => (
        <div className="space-y-6 animate-fade-in text-cyan-200 h-full overflow-y-auto custom-scrollbar pr-2">
            <div className="bg-gradient-to-r from-cyan-900/40 to-blue-900/40 p-6 rounded-xl border border-cyan-500/30 flex flex-col md:flex-row items-center gap-6 relative overflow-hidden group">
                <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
                
                <div className="relative w-32 h-32 flex-shrink-0 transition-transform duration-700 group-hover:scale-105">
                    <div className="absolute inset-0 bg-cyan-500/20 blur-xl rounded-full animate-pulse"></div>
                    <svg viewBox="0 0 100 100" className="w-full h-full drop-shadow-[0_0_15px_rgba(6,182,212,0.8)]">
                        <polygon points="50 2, 93 25, 93 75, 50 98, 7 75, 7 25" fill="rgba(0,10,20,0.9)" stroke="#22d3ee" strokeWidth="2" />
                        <path d="M 50 20 L 50 35 M 50 80 L 50 65 M 20 50 L 35 50 M 80 50 L 65 50" stroke="#0891b2" strokeWidth="2" />
                        <circle cx="50" cy="50" r="15" fill="none" stroke="#22d3ee" strokeWidth="1.5" className="animate-spin-slow" style={{transformOrigin: '50% 50%'}} />
                        <text x="50" y="55" fontSize="14" fill="white" textAnchor="middle" fontFamily="monospace" fontWeight="bold">CYC</text>
                    </svg>
                </div>
    
                <div className="flex-grow text-center md:text-left z-10">
                    <h2 className="text-3xl font-bold text-white tracking-widest mb-1">CyChips <span className="text-cyan-400 text-lg align-top font-mono">CYC</span></h2>
                    <p className="text-lg text-cyan-300 font-light italic">"The People’s Bargaining Chip."</p>
                    <p className="text-xs text-gray-400 mt-2 max-w-md">Don't just watch the game. Own the chips. Decentralized wealth, one chip at a time.</p>
                </div>
            </div>
    
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-black/30 p-4 rounded-lg border border-cyan-800/50 hover:border-cyan-600 transition-colors">
                    <h4 className="text-cyan-400 font-bold mb-2 flex items-center text-sm uppercase tracking-wider"><UsersIcon className="w-4 h-4 mr-2"/> Micro-Equity</h4>
                    <p className="text-xs text-gray-300 leading-relaxed">
                        Own a seat at the table. Unlike traditional high-priced assets, CyChips allows you to hold tangible, fractional value. Wealth in small bytes.
                    </p>
                </div>
                <div className="bg-black/30 p-4 rounded-lg border border-cyan-800/50 hover:border-cyan-600 transition-colors">
                    <h4 className="text-cyan-400 font-bold mb-2 flex items-center text-sm uppercase tracking-wider"><ActivityIcon className="w-4 h-4 mr-2"/> Proof of Participation</h4>
                    <p className="text-xs text-gray-300 leading-relaxed">
                        Legitimo Value. Backed by the work put into the CHIPS network. Earned through decentralized computation, data sharing, and community governance.
                    </p>
                </div>
            </div>
    
            <div className="bg-black/20 p-4 rounded-lg border border-cyan-900/50">
                <div className="flex items-center justify-between mb-4 border-b border-cyan-800/50 pb-2">
                    <h3 className="text-sm font-bold text-white">Network Utility & Market</h3>
                    <span className="text-[10px] text-green-400 flex items-center"><div className="w-2 h-2 rounded-full bg-green-500 mr-1 animate-pulse"></div> Live Market</span>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="text-center p-2 bg-cyan-950/20 rounded">
                        <p className="text-[10px] text-gray-500 uppercase">Current Price</p>
                        <p className="text-xl font-mono text-white">${cyChipsData.price}</p>
                    </div>
                    <div className="text-center p-2 bg-cyan-950/20 rounded">
                        <p className="text-[10px] text-gray-500 uppercase">Market Cap</p>
                        <p className="text-xl font-mono text-white">{cyChipsData.marketCap}</p>
                    </div>
                    <div className="text-center p-2 bg-cyan-950/20 rounded">
                        <p className="text-[10px] text-gray-500 uppercase">24h Volume</p>
                        <p className="text-xl font-mono text-white">{cyChipsData.volume}</p>
                    </div>
                    <div className="text-center p-2 bg-cyan-950/20 rounded">
                        <p className="text-[10px] text-gray-500 uppercase">Growth</p>
                        <p className="text-xl font-mono text-green-400">{cyChipsData.change}</p>
                    </div>
                </div>
    
                <div className="bg-cyan-950/20 p-3 rounded border border-cyan-800/30">
                    <p className="text-xs text-cyan-300 font-bold mb-2">Exclusive Network Currency For:</p>
                    <div className="flex flex-wrap gap-2">
                        {['App Deployment', 'Domain Registration', 'Hosting Fees', 'QPU Time', 'System Upgrades'].map(tag => (
                            <span key={tag} className="px-2 py-1 rounded bg-cyan-900/40 border border-cyan-700/50 text-[10px] text-cyan-200 flex items-center hover:bg-cyan-800/50 cursor-default">
                                <CpuChipIcon className="w-3 h-3 mr-1 opacity-70" /> {tag}
                            </span>
                        ))}
                    </div>
                </div>
            </div>

            <CyChipsEncryptionModule />
        </div>
    );

    return (
        <GlassPanel title={
            <div className="flex items-center">
                <BanknotesIcon className="w-5 h-5 mr-2 text-green-400" />
                <span>ChipsEconomy Central</span>
            </div>
        }>
            <div className="flex flex-col h-full animate-fade-in overflow-hidden">
                <div className="flex space-x-1 mb-4 bg-black/20 p-1 rounded-lg border border-cyan-900/50 w-fit mx-auto overflow-x-auto no-scrollbar flex-shrink-0">
                    <button 
                        onClick={() => setActiveTab('rewards')}
                        className={`px-4 py-1.5 text-xs font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'rewards' ? 'bg-amber-600 text-white shadow-lg shadow-amber-900/50' : 'text-cyan-500 hover:bg-white/5'}`}
                    >
                        <StarIcon className="w-4 h-4" /> Rewards Matrix
                    </button>
                    <button 
                        onClick={() => setActiveTab('cychips')}
                        className={`px-4 py-1.5 text-xs font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'cychips' ? 'bg-cyan-600 text-white shadow-lg' : 'text-cyan-500 hover:bg-white/5'}`}
                    >
                        <CpuChipIcon className="w-4 h-4" /> CyChips Coin
                    </button>
                    <div className="w-px h-6 bg-cyan-900/50 mx-1"></div>
                    <button 
                        onClick={() => setActiveTab('currency')}
                        className={`px-4 py-1.5 text-xs font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'currency' ? 'bg-green-600/40 text-white border border-green-500/50' : 'text-cyan-500 hover:bg-white/5'}`}
                    >
                        <BanknotesIcon className="w-4 h-4" /> Legacy Currency
                    </button>
                    <button 
                        onClick={() => setActiveTab('wallet')}
                        className={`px-4 py-1.5 text-xs font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'wallet' ? 'bg-blue-600/40 text-white border border-blue-500/50' : 'text-cyan-500 hover:bg-white/5'}`}
                    >
                        <WalletIcon className="w-4 h-4" /> Wallet & Payments
                    </button>
                    <button 
                        onClick={() => setActiveTab('exchange')}
                        className={`px-4 py-1.5 text-xs font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'exchange' ? 'bg-purple-600/40 text-white border border-purple-500/50' : 'text-cyan-500 hover:bg-white/5'}`}
                    >
                        <RefreshCwIcon className="w-4 h-4" /> Q-DEX Exchange
                    </button>
                    <button 
                        onClick={() => setActiveTab('legal')}
                        className={`px-4 py-1.5 text-xs font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'legal' ? 'bg-slate-600/60 text-white border border-slate-500/50' : 'text-cyan-500 hover:bg-white/5'}`}
                    >
                        <ScaleIcon className="w-4 h-4" /> Legalities
                    </button>
                </div>

                <div className="flex-grow min-h-0 overflow-hidden relative">
                    {activeTab === 'rewards' && <CyChipsRewardPanel />}
                    {activeTab === 'cychips' && renderCyChips()}
                    {activeTab === 'legal' && renderLegalities()}
                    {activeTab === 'currency' && (
                        <div className="space-y-4 animate-fade-in h-full overflow-y-auto custom-scrollbar">
                            <div className="bg-black/20 p-4 rounded-lg border border-green-800/50">
                                <h3 className="text-sm font-bold text-green-300 mb-4 flex items-center">
                                    <ShieldCheckIcon className="w-4 h-4 mr-2" /> Digital Currency Back Office
                                </h3>
                                <div className="grid grid-cols-2 gap-4 text-xs">
                                    <div className="bg-green-900/20 p-3 rounded border border-green-700/30">
                                        <p className="text-green-500 uppercase tracking-widest mb-1">Token Name</p>
                                        <p className="text-xl text-white font-mono">{currencyData.name} ({currencyData.symbol})</p>
                                    </div>
                                    <div className="bg-green-900/20 p-3 rounded border border-green-700/30">
                                        <p className="text-green-500 uppercase tracking-widest mb-1">AGI Stability Index</p>
                                        <p className="text-xl text-white font-mono">{currencyData.stabilityIndex}</p>
                                    </div>
                                    <div className="bg-black/40 p-3 rounded border border-cyan-900/30">
                                        <p className="text-gray-400 uppercase tracking-widest mb-1">Circulating Supply</p>
                                        <p className="text-lg text-cyan-200 font-mono">{currencyData.circulating}</p>
                                    </div>
                                    <div className="bg-black/40 p-3 rounded border border-cyan-900/30">
                                        <p className="text-gray-400 uppercase tracking-widest mb-1">Total Supply</p>
                                        <p className="text-lg text-cyan-200 font-mono">{currencyData.totalSupply}</p>
                                    </div>
                                </div>
                                <div className="mt-4 p-3 bg-black/40 rounded border border-cyan-900/30 flex justify-between items-center">
                                    <span className="text-xs text-cyan-400">Current Minting Rate</span>
                                    <span className="text-sm font-bold text-white font-mono animate-pulse">{currencyData.mintingRate}</span>
                                </div>
                            </div>
                        </div>
                    )}
                    {activeTab === 'wallet' && (
                        <div className="space-y-4 animate-fade-in h-full overflow-y-auto custom-scrollbar">
                            <div className="bg-black/20 p-4 rounded-lg border border-blue-800/50">
                                <h3 className="text-sm font-bold text-blue-300 mb-4 flex items-center">
                                    <LockIcon className="w-4 h-4 mr-2" /> ChipsPay Cashier & Wallet Ops
                                </h3>
                                
                                <div className="flex items-center justify-between bg-blue-900/20 p-3 rounded border border-blue-700/30 mb-4">
                                    <div className="flex items-center gap-2">
                                        <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_#22c55e]"></div>
                                        <span className="text-xs font-bold text-blue-200">Gateway Status</span>
                                    </div>
                                    <span className="text-xs font-mono text-green-400">{walletData.gatewayStatus}</span>
                                </div>

                                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs mb-4">
                                    <div className="text-center p-2 bg-black/40 rounded border border-cyan-900/30">
                                        <p className="text-gray-500 mb-1">Throughput</p>
                                        <p className="text-lg text-white font-mono">{walletData.throughput}</p>
                                    </div>
                                    <div className="text-center p-2 bg-black/40 rounded border border-cyan-900/30">
                                        <p className="text-gray-500 mb-1">Escrow Pool</p>
                                        <p className="text-lg text-white font-mono">{walletData.escrowVolume}</p>
                                    </div>
                                    <div className="text-center p-2 bg-black/40 rounded border border-cyan-900/30">
                                        <p className="text-gray-500 mb-1">Fraud Alerts</p>
                                        <p className="text-lg text-green-400 font-mono">{walletData.fraudChecks}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                    {activeTab === 'exchange' && (
                        <div className="space-y-4 animate-fade-in h-full overflow-y-auto custom-scrollbar">
                            <div className="bg-black/20 p-4 rounded-lg border border-purple-800/50">
                                <h3 className="text-sm font-bold text-purple-300 mb-4 flex items-center">
                                    <ChartBarIcon className="w-4 h-4 mr-2" /> Q-DEX Money Exchange
                                </h3>
                                <div className="flex items-center gap-4 mb-4">
                                    <div className="flex-1 bg-purple-900/20 p-3 rounded border border-purple-700/30 text-center">
                                        <p className="text-[10px] text-purple-400 uppercase">Primary Pair</p>
                                        <p className="text-lg font-bold text-white mt-1">{exchangeData.pair}</p>
                                    </div>
                                    <div className="flex-1 bg-purple-900/20 p-3 rounded border border-purple-700/30 text-center">
                                        <p className="text-[10px] text-purple-400 uppercase">Exchange Rate</p>
                                        <p className="text-lg font-bold text-white mt-1">{exchangeData.rate}</p>
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <div className="flex justify-between items-center bg-black/40 p-2 rounded border border-cyan-900/30">
                                        <span className="text-xs text-gray-400 flex items-center gap-2"><ActivityIcon className="w-3 h-3"/> Liquidity Depth</span>
                                        <span className="text-xs text-green-400 font-mono">{exchangeData.liquidity}</span>
                                    </div>
                                    <div className="flex justify-between items-center bg-black/40 p-2 rounded border border-cyan-900/30">
                                        <span className="text-xs text-gray-400 flex items-center gap-2"><RefreshCwIcon className="w-3 h-3"/> 24h Volume</span>
                                        <span className="text-xs text-cyan-300 font-mono">{exchangeData.volume24h}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </GlassPanel>
    );
};

export default ChipsEconomy;
