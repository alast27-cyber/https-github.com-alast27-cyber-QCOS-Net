
import React, { useState, useEffect } from 'react';
import { 
    ShieldCheckIcon, UsersIcon, LockIcon, CheckCircle2Icon, 
    PlusIcon, TrashIcon, SaveIcon, ServerCogIcon, 
    CpuChipIcon, GlobeIcon, CodeBracketIcon, BanknotesIcon 
} from './Icons';

interface Permission {
    id: string;
    label: string;
    description: string;
}

interface PermissionCategory {
    id: string;
    label: string;
    icon: React.FC<{className?: string}>;
    permissions: Permission[];
}

interface Role {
    id: string;
    name: string;
    description: string;
    permissions: string[];
    isSystem?: boolean;
    color: string;
}

const PERMISSION_SCHEMA: PermissionCategory[] = [
    {
        id: 'dashboard',
        label: 'Dashboard Access',
        icon: GlobeIcon,
        permissions: [
            { id: 'view_core', label: 'View Core System', description: 'Access to Agent Q Core and System Vitals' },
            { id: 'view_finance', label: 'View Economy', description: 'Access to Chips Economy and Q-DEX' },
            { id: 'view_dev', label: 'View Dev Platform', description: 'Access to ChipsDev Platform' },
            { id: 'view_security', label: 'View Security', description: 'Access to Security Monitor' },
        ]
    },
    {
        id: 'agent_q',
        label: 'Agent Q Modules',
        icon: CpuChipIcon,
        permissions: [
            { id: 'interact_neural', label: 'Neural Interface', description: 'Can inject intents via Neural Bridge' },
            { id: 'run_simulation', label: 'Run Simulations', description: 'Can trigger Universe/QNN Simulations' },
            { id: 'train_models', label: 'Train Models', description: 'Can initiate QML training sessions' },
            { id: 'access_thoughts', label: 'Read Thoughts', description: 'View raw internal Agent Q logs' },
        ]
    },
    {
        id: 'dev_ops',
        label: 'Development & Ops',
        icon: CodeBracketIcon,
        permissions: [
            { id: 'deploy_prod', label: 'Deploy to Production', description: 'Push apps to public gateways' },
            { id: 'edit_code', label: 'Edit Source Code', description: 'Modify system kernel files' },
            { id: 'approve_apps', label: 'Approve Store Apps', description: 'Review and approve App Exchange submissions' },
        ]
    },
    {
        id: 'admin',
        label: 'System Admin',
        icon: ServerCogIcon,
        permissions: [
            { id: 'manage_users', label: 'Manage Users', description: 'Create/Delete users and assign roles' },
            { id: 'manage_roles', label: 'Manage Roles', description: 'Modify permission sets (This Panel)' },
            { id: 'system_override', label: 'System Override', description: 'Emergency shutdown and kernel reset' },
        ]
    }
];

const INITIAL_ROLES: Role[] = [
    {
        id: 'admin',
        name: 'Administrator',
        description: 'Full system access and control.',
        permissions: ['view_core', 'view_finance', 'view_dev', 'view_security', 'interact_neural', 'run_simulation', 'train_models', 'access_thoughts', 'deploy_prod', 'edit_code', 'approve_apps', 'manage_users', 'manage_roles', 'system_override'],
        isSystem: true,
        color: 'text-red-400'
    },
    {
        id: 'dev',
        name: 'Developer',
        description: 'Can build, test, and simulate applications.',
        permissions: ['view_dev', 'view_core', 'run_simulation', 'edit_code', 'train_models'],
        isSystem: false,
        color: 'text-yellow-400'
    },
    {
        id: 'operator',
        name: 'Operator',
        description: 'Monitors system health and security.',
        permissions: ['view_core', 'view_security', 'view_finance', 'run_simulation'],
        isSystem: false,
        color: 'text-green-400'
    },
    {
        id: 'viewer',
        name: 'Observer',
        description: 'Read-only access to dashboards.',
        permissions: ['view_core', 'view_finance'],
        isSystem: false,
        color: 'text-blue-400'
    }
];

const RolePermissionsPanel: React.FC = () => {
    const [roles, setRoles] = useState<Role[]>(() => {
        const saved = localStorage.getItem('qcos_roles');
        return saved ? JSON.parse(saved) : INITIAL_ROLES;
    });
    const [selectedRoleId, setSelectedRoleId] = useState<string>('admin');
    const [isSaving, setIsSaving] = useState(false);
    const [isCreating, setIsCreating] = useState(false);
    const [newRoleName, setNewRoleName] = useState('');

    useEffect(() => {
        localStorage.setItem('qcos_roles', JSON.stringify(roles));
    }, [roles]);

    const selectedRole = roles.find(r => r.id === selectedRoleId) || roles[0];

    const togglePermission = (permId: string) => {
        if (selectedRole.isSystem && selectedRole.id === 'admin') return; // Admin is immutable

        setRoles(prev => prev.map(role => {
            if (role.id === selectedRoleId) {
                const hasPerm = role.permissions.includes(permId);
                return {
                    ...role,
                    permissions: hasPerm 
                        ? role.permissions.filter(p => p !== permId)
                        : [...role.permissions, permId]
                };
            }
            return role;
        }));
    };

    const handleCreateRole = () => {
        if (!newRoleName.trim()) return;
        const newRole: Role = {
            id: newRoleName.toLowerCase().replace(/\s+/g, '_'),
            name: newRoleName,
            description: 'Custom user defined role.',
            permissions: [],
            isSystem: false,
            color: 'text-purple-400'
        };
        setRoles([...roles, newRole]);
        setSelectedRoleId(newRole.id);
        setNewRoleName('');
        setIsCreating(false);
    };

    const handleDeleteRole = (roleId: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (window.confirm('Are you sure you want to delete this role?')) {
            setRoles(roles.filter(r => r.id !== roleId));
            if (selectedRoleId === roleId) setSelectedRoleId(roles[0].id);
        }
    };

    const handleSave = () => {
        setIsSaving(true);
        setTimeout(() => setIsSaving(false), 1000);
    };

    return (
        <div className="h-full flex flex-col md:flex-row gap-4 animate-fade-in text-cyan-200">
            
            {/* Left Column: Role List */}
            <div className="w-full md:w-1/3 bg-black/30 rounded-lg border border-cyan-900/50 flex flex-col overflow-hidden">
                <div className="p-3 bg-cyan-950/30 border-b border-cyan-900/50 flex justify-between items-center">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                        <UsersIcon className="w-4 h-4" /> Roles
                    </h3>
                    <button 
                        onClick={() => setIsCreating(true)} 
                        className="p-1 rounded hover:bg-white/10 text-cyan-400 hover:text-white transition-colors"
                        title="Create New Role"
                    >
                        <PlusIcon className="w-4 h-4" />
                    </button>
                </div>
                
                <div className="flex-grow overflow-y-auto p-2 space-y-1">
                    {isCreating && (
                        <div className="p-2 bg-black/50 rounded border border-cyan-700 flex flex-col gap-2 mb-2 animate-fade-in">
                            <input 
                                autoFocus
                                type="text" 
                                placeholder="Role Name..." 
                                value={newRoleName}
                                onChange={(e) => setNewRoleName(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleCreateRole()}
                                className="w-full bg-black/30 border border-cyan-800 rounded px-2 py-1 text-xs text-white focus:border-cyan-500 outline-none"
                            />
                            <div className="flex gap-2">
                                <button onClick={handleCreateRole} className="flex-1 bg-green-900/30 text-green-400 text-[10px] py-1 rounded hover:bg-green-900/50">Create</button>
                                <button onClick={() => setIsCreating(false)} className="flex-1 bg-red-900/30 text-red-400 text-[10px] py-1 rounded hover:bg-red-900/50">Cancel</button>
                            </div>
                        </div>
                    )}

                    {roles.map(role => (
                        <button
                            key={role.id}
                            onClick={() => setSelectedRoleId(role.id)}
                            className={`w-full text-left p-3 rounded-lg border transition-all duration-200 group relative ${
                                selectedRoleId === role.id 
                                    ? 'bg-cyan-900/40 border-cyan-500/50 shadow-[0_0_10px_rgba(0,255,255,0.1)]' 
                                    : 'bg-transparent border-transparent hover:bg-white/5 hover:border-white/10'
                            }`}
                        >
                            <div className="flex justify-between items-start">
                                <div>
                                    <div className={`font-bold text-sm ${role.color}`}>{role.name}</div>
                                    <div className="text-[10px] text-gray-500 mt-0.5">{role.permissions.length} permissions</div>
                                </div>
                                {role.isSystem && <span title="System Role"><LockIcon className="w-3 h-3 text-gray-600" /></span>}
                                {!role.isSystem && (
                                    <div 
                                        onClick={(e) => handleDeleteRole(role.id, e)}
                                        className="opacity-0 group-hover:opacity-100 p-1 text-red-500 hover:text-red-300 hover:bg-red-900/20 rounded transition-all"
                                    >
                                        <TrashIcon className="w-3 h-3" />
                                    </div>
                                )}
                            </div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Right Column: Permission Matrix */}
            <div className="flex-grow bg-black/30 rounded-lg border border-cyan-900/50 flex flex-col overflow-hidden">
                <div className="p-3 bg-cyan-950/30 border-b border-cyan-900/50 flex justify-between items-center">
                    <div className="flex flex-col">
                        <h3 className="text-sm font-bold text-white flex items-center gap-2">
                            <ShieldCheckIcon className="w-4 h-4" /> Permissions Matrix
                        </h3>
                        <div className="text-[10px] text-cyan-500/70">Configure access levels for {roles.find(r => r.id === selectedRoleId)?.name}</div>
                    </div>
                    <button 
                        onClick={handleSave}
                        disabled={isSaving}
                        className={`px-3 py-1 rounded text-xs font-bold transition-all ${
                            isSaving ? 'bg-gray-700 text-gray-400' : 'bg-cyan-600 text-white hover:bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                        }`}
                    >
                        {isSaving ? 'Saving...' : 'Save Changes'}
                    </button>
                </div>

                <div className="flex-grow overflow-y-auto p-4 custom-scrollbar">
                    <div className="space-y-6">
                        {PERMISSION_SCHEMA.map(category => (
                            <div key={category.id} className="space-y-3">
                                <div className="flex items-center gap-2 text-cyan-400 border-b border-cyan-900/30 pb-1">
                                    <category.icon className="w-3 h-3" />
                                    <span className="text-[10px] font-bold uppercase tracking-wider">{category.label}</span>
                                </div>
                                <div className="grid grid-cols-1 gap-2">
                                    {category.permissions.map(perm => (
                                        <div key={perm.id} className="flex items-center justify-between p-2 bg-white/5 rounded border border-white/5 hover:border-cyan-900/30 transition-colors">
                                            <div className="flex flex-col">
                                                <span className="text-xs font-medium text-white">{perm.label}</span>
                                                <span className="text-[9px] text-gray-500">{perm.description}</span>
                                            </div>
                                            <button 
                                                onClick={() => togglePermission(perm.id)}
                                                className={`w-8 h-4 rounded-full relative transition-colors ${
                                                    roles.find(r => r.id === selectedRoleId)?.permissions.includes(perm.id)
                                                        ? 'bg-cyan-600'
                                                        : 'bg-gray-700'
                                                }`}
                                            >
                                                <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all ${
                                                    roles.find(r => r.id === selectedRoleId)?.permissions.includes(perm.id)
                                                        ? 'left-4.5'
                                                        : 'left-0.5'
                                                }`} />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default RolePermissionsPanel;
