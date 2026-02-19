import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { BookOpenIcon, KeyIcon, LockIcon, AlertTriangleIcon, LoaderIcon, EyeIcon, EyeOffIcon, ShieldCheckIcon, RefreshCwIcon } from './Icons';
import { useAuth } from '../context/AuthContext';
import { userService } from '../services/userService';

const LoginScreen: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loginMessage, setLoginMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const { login } = useAuth();

  const handleLogin = async () => {
    if (!username || !password) {
        setLoginMessage('Please enter username and password.');
        return;
    }

    setLoginMessage('Authenticating...');
    setIsProcessing(true);

    await new Promise(resolve => setTimeout(resolve, 800));
    
    try {
        const user = userService.authenticate(username, password);

        if (user) {
            setLoginMessage('Login successful! Initializing dashboard...');
            setTimeout(() => {
                login(user.username, user.level);
            }, 500);
        } else {
            setLoginMessage('Invalid username or password.');
            setIsProcessing(false);
        }
    } catch (e: any) {
        setLoginMessage(e.message || 'Authentication error.');
        setIsProcessing(false);
    }
  };
  
   const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleLogin();
    }
  };

  const handleFactoryReset = () => {
      if (window.confirm("WARNING: This will wipe local QCOS nodes and force a fresh 'Ground Up' installation of ChipsBrowser. Continue?")) {
          localStorage.removeItem('chips_browser_installed');
          localStorage.removeItem('qcos_onboarded');
          window.location.reload();
      }
  };

  return (
    <GlassPanel title='QCOS Gateway Access'>
      <div className="flex flex-col space-y-8 p-4 text-cyan-200 h-full overflow-y-auto relative">
        <div className="absolute top-0 right-4 bg-green-900/30 border border-green-500/50 rounded-full px-3 py-1 flex items-center gap-2 text-xs text-green-300">
            <ShieldCheckIcon className="w-3 h-3" />
            <span>Server: 172.16.1.170 (Secure)</span>
        </div>

        <div className="flex items-start space-x-3">
          <BookOpenIcon className="w-8 h-8 flex-shrink-0 text-cyan-400" />
          <div>
            <h3 className="text-xl font-semibold text-cyan-300 mb-2">About QCOS</h3>
            <p className="text-sm leading-relaxed">
              The Quantum Computing Operating System (QCOS) is a platform for interacting with quantum systems, powered by the Agent Q QNN core. It provides tools for quantum programming (Q-Lang), application deployment, and real-time monitoring of quantum hardware.
            </p>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-cyan-800 flex items-start space-x-3">
          <LockIcon className="w-8 h-8 flex-shrink-0 text-cyan-400" />
          <div className="flex-1">
            <h3 className="text-xl font-semibold text-cyan-300 mb-4">Operator Login</h3>
            <div className="space-y-4 max-w-sm">
              <div>
                <label className="block text-sm font-medium text-cyan-300 mb-1" htmlFor="username">Username</label>
                <input
                  type="text"
                  id="username"
                  className="w-full p-2 bg-black/30 border border-cyan-700 rounded-md focus:outline-none focus:ring-2 focus:ring-cyan-500 text-cyan-100 placeholder-cyan-400"
                  placeholder="Enter username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  onKeyDown={handleKeyDown}
                  autoComplete="username"
                  disabled={isProcessing}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-cyan-300 mb-1" htmlFor="password">Password</label>
                <div className="relative">
                    <input
                      type={showPassword ? 'text' : 'password'}
                      id="password"
                      className="w-full p-2 bg-black/30 border border-cyan-700 rounded-md focus:outline-none focus:ring-2 focus:ring-cyan-500 text-cyan-100 placeholder-cyan-400 pr-10"
                      placeholder="Enter password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      onKeyDown={handleKeyDown}
                      autoComplete="current-password"
                      disabled={isProcessing}
                    />
                    <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute inset-y-0 right-0 flex items-center px-3 text-cyan-500 hover:text-cyan-300 focus:outline-none"
                        tabIndex={-1}
                    >
                        {showPassword ? <EyeOffIcon className="w-4 h-4" /> : <EyeIcon className="w-4 h-4" />}
                    </button>
                </div>
              </div>
              <button
                onClick={handleLogin}
                disabled={isProcessing}
                title="Login to the QCOS network with your credentials."
                className={`w-full py-2 px-4 bg-cyan-600/50 hover:bg-cyan-700/70 text-white font-bold rounded-md transition duration-200 holographic-button flex items-center justify-center space-x-2 ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {isProcessing ? <LoaderIcon className="w-5 h-5 animate-spin" /> : <KeyIcon className="w-5 h-5" />}
                <span>{isProcessing ? 'Verifying Identity...' : 'Login to Dashboard'}</span>
              </button>
              {loginMessage && (
                <>
                    {loginMessage.includes('Invalid') || loginMessage.includes('error') ? (
                        <div className="flex items-start gap-2 text-red-300 bg-red-900/50 p-3 rounded-md border border-red-700 text-sm animate-fade-in">
                            <AlertTriangleIcon className="w-5 h-5 flex-shrink-0 mt-0.5"/>
                            <span>{loginMessage}</span>
                        </div>
                    ) : (
                        <div className={`flex items-center justify-center gap-2 text-sm animate-fade-in ${loginMessage.includes('successful') ? 'text-green-400' : 'text-cyan-300'}`}>
                            {loginMessage.includes('Authenticating') && <LoaderIcon className="w-4 h-4 animate-spin"/>}
                            <span>{loginMessage}</span>
                        </div>
                    )}
                </>
              )}
            </div>
            
            <div className="mt-6 text-xs text-gray-500 border-t border-cyan-900/30 pt-2">
                <p>Default Accounts:</p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-1 font-mono text-cyan-600/70">
                    <span>admin / qcos@123</span>
                    <span>dev / dev</span>
                    <span>op / op</span>
                    <span>guest / guest</span>
                </div>
            </div>

            <div className="mt-8 flex justify-center">
                <button 
                    onClick={handleFactoryReset}
                    className="text-[10px] text-red-400 hover:text-red-300 flex items-center gap-1 border border-red-900/50 px-3 py-1 rounded bg-red-950/20 hover:bg-red-900/40 transition-colors"
                >
                    <RefreshCwIcon className="w-3 h-3" />
                    [ ! ] RESET SIMULATION & RE-INSTALL
                </button>
            </div>
          </div>
        </div>

      </div>
    </GlassPanel>
  );
};

export default LoginScreen;